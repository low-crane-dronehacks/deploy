#!/usr/bin/env python3
"""Start, inspect and stop YOUR onboard ROS processes. Never arms or flies.

    python3 onboard/services.py start
    python3 onboard/services.py status
    python3 onboard/services.py stop

The process list is ordinary editable JSON. PID identity, source receipts and
logs are under .onboard/. No cloud API, Docker socket or simulator source is used.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import time

SCHEMA = "flyx-onboard-processes/1"


def project_root(value=None):
    root = Path(value or Path(__file__).resolve().parent.parent).absolute()
    if root.is_symlink() or not (root / "onboard/config/services.json").is_file():
        raise ValueError("select a project containing onboard/config/services.json")
    return root.resolve()


def source_hashes(root):
    result = {}
    for p in sorted((root / "onboard").rglob("*")):
        if "__pycache__" in p.parts:
            continue
        if p.is_symlink():
            raise ValueError("onboard source must contain regular files, not symlinks")
        if p.is_file():
            result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if not result:
        raise ValueError("onboard source is missing")
    return result


def starttime(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().split(") ", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def atomic(path, value):
    fd, name = tempfile.mkstemp(prefix=".onboard-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


@contextlib.contextmanager
def registry(root):
    folder = root / ".onboard"
    if folder.is_symlink():
        raise ValueError("onboard state must not be a symlink")
    folder.mkdir(mode=0o700, exist_ok=True)
    lock = folder / "lock"
    if lock.is_symlink():
        raise ValueError("onboard lock must not be a symlink")
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = folder / "state.json"
        if state.is_symlink():
            raise ValueError("onboard registry must not be a symlink")
        value = json.loads(state.read_text()) if state.exists() else None
        if value is not None and (value.get("schema") != SCHEMA or value.get("project") != str(root)
                                  or not re.fullmatch(r"[0-9a-f]{32}", value.get("token", ""))):
            raise ValueError("onboard registry identity is invalid; inspect it before recovery")
        yield folder, state, value


def owned(record):
    """All descendants carrying this launch's random token, never process names."""
    marker = ("FLYX_ONBOARD_TOKEN=" + record["token"]).encode()
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            if marker not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            pid = int(directory.name); stamp = starttime(pid)
            if stamp is not None:
                result[pid] = stamp
        except (OSError, ValueError):
            continue
    return result


def view(root, record):
    if record is None:
        return {"state": "stopped", "processes": [], "project": str(root)}
    alive = owned(record)
    processes = [dict(p, alive=alive.get(p["pid"]) == p["start"]) for p in record.get("processes", [])]
    live = bool(processes) and all(p["alive"] for p in processes)
    return {**record, "state": "running" if live else "incomplete" if alive else "stopped",
            "processes": processes, "owned_pids": sorted(alive),
            "source_matches": source_hashes(root) == record["sources"]}


def status(root):
    with registry(root) as (_, _, record):
        return view(root, record)


def stop_owned(record, grace_s=5.0):
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 2.0)):
        for pid, stamp in owned(record).items():
            if starttime(pid) == stamp:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, sig)
        until = time.monotonic() + wait
        while owned(record) and time.monotonic() < until:
            time.sleep(0.05)
        if not owned(record):
            return
    raise RuntimeError("onboard processes survived termination; ownership remains recorded")


def stop(root, token=None):
    with registry(root) as (_, path, record):
        if record is None:
            return {"state": "stopped", "project": str(root), "processes": []}
        if token is not None and record["token"] != token:
            raise ValueError("another onboard launch owns these processes; left untouched")
        stop_owned(record)
        record["ended_at"] = time.time(); atomic(path, record)
        atomic(path.parent / record["token"] / "launch.json", record)
        return view(root, record)


def plan(root, mode, deployment=None):
    config = json.loads((root / "onboard/config/services.json").read_text())
    if config.get("schema") != "flyx-onboard-services/1" or mode not in config.get("profiles", {}):
        raise ValueError("select a declared onboard service profile")
    rows = []
    for item in config["profiles"][mode]:
        name, argv = item.get("name", ""), item.get("argv")
        if (not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", name) or not isinstance(argv, list)
                or not argv or any(not isinstance(a, str) or "\0" in a for a in argv)):
            raise ValueError("service needs a name and ordinary argv list")
        argv = [a.replace("{project}", str(root)) for a in argv]
        if deployment is not None:
            argv += ["--params-file", str(deployment)]
        rows.append({"name": name, "argv": argv})
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("service names must be nonempty and unique")
    return rows


def start(root, mode="vio", deployment=None, token=None, world_token=None):
    if deployment is not None:
        deployment = Path(deployment).resolve(strict=True)
    rows = plan(root, mode, deployment)
    with registry(root) as (folder, path, previous):
        if previous is not None and owned(previous):
            raise ValueError("onboard processes are already running; stop the recorded launch first")
        token = token or secrets.token_hex(16)
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("onboard launch requires a random 32-character hex token")
        run = folder / token
        run.mkdir(mode=0o700)
        sources = source_hashes(root)
        # Preserve the exact starting source for review, without pretending a
        # later workspace edit changed an already-running Python process.
        for name in sources:
            target = run / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        record = {"schema": SCHEMA, "token": token, "project": str(root), "mode": mode,
                  "world_token": world_token,
                  "created_at": time.time(), "sources": sources, "processes": [],
                  "deployment": None if deployment is None else {"path": str(deployment),
                      "sha256": hashlib.sha256(deployment.read_bytes()).hexdigest()}}
        atomic(path, record)  # token recorded BEFORE any child exists
        atomic(run / "launch.json", record)
        env = dict(os.environ, FLYX_ONBOARD_TOKEN=token,
                   PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        try:
            for row in rows:
                log_path = run / (row["name"] + ".log")
                with log_path.open("ab", buffering=0) as log:
                    child = subprocess.Popen(row["argv"], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stamp = starttime(child.pid)
                if stamp is None:
                    raise RuntimeError(row["name"] + " exited at startup; inspect " + str(log_path))
                record["processes"].append(dict(row, pid=child.pid, start=stamp, log=str(log_path)))
                atomic(path, record)
                atomic(run / "launch.json", record)
            time.sleep(0.15)
            result = view(root, record)
            if result["state"] != "running":
                raise RuntimeError("an onboard service exited; inspect the recorded logs")
            return result
        except BaseException:
            stop_owned(record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop"))
    parser.add_argument("--project", type=Path)
    parser.add_argument("--mode", choices=("vio", "gnss"))
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--token", help="exact launch ownership, for lifecycle integration")
    args = parser.parse_args(argv)
    root = project_root(args.project)
    deployment = args.deployment
    mode = args.mode
    world_token = None
    metadata = Path("/etc/flyx/onboard.json")
    if args.command == "start" and metadata.is_file():
        info = json.loads(metadata.read_text()); deployment = deployment or Path(info["params_file"])
        if info.get("onboard_owner") != "companion":
            parser.error("the simulator has not handed estimation to this workspace")
        mode = mode or info["mode"]
        world_token = info.get("world_token")
    if args.command == "start" and os.environ.get("FLYX_LAB_ACTIVE") == "1" and not metadata.is_file():
        parser.error("this worker still owns its VIO externally; explicit workspace migration is required")
    try:
        result = (start(root, mode or "vio", deployment, args.token, world_token) if args.command == "start" else
                  stop(root, args.token) if args.command == "stop" else status(root))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, "onboard: " + str(error) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
