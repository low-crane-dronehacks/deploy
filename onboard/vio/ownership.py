"""One local vision publisher, plus a bounded DDS authority discovery window."""
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import time


class PublisherGuard:
    def __init__(self,node,topic,window_s=25.0):
        self.node,self.topic=node,topic
        self.window_s=max(25.,float(window_s));self.started=time.monotonic()
        self.counts=[];self.ready=False
        tag=hashlib.sha256((os.environ.get('ROS_DOMAIN_ID','0')+':'+topic).encode()).hexdigest()[:20]
        path=Path(tempfile.gettempdir())/f'px4-vision-{os.getuid()}-{tag}.lock'
        fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
        self.stream=os.fdopen(fd,'r+')
        info=os.fstat(fd)
        if info.st_uid!=os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_mode&0o077:
            self.close();raise ValueError('vision ownership file is not private and regular')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.close();raise ValueError('another local process owns this vision input') from None

    def may_publish(self):
        if self.ready:return True
        try:count=self.node.count_publishers(self.topic)
        except Exception:count=None
        self.counts.append(count)
        if time.monotonic()-self.started<self.window_s:return False
        if not self.counts or any(c is None for c in self.counts) or max(self.counts)!=1:
            raise RuntimeError('cannot prove sole vision publisher during the 25-second discovery window')
        self.ready=True
        return True

    def close(self):
        if self.stream is not None:
            self.stream.close();self.stream=None
