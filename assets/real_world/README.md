# Real-world course model (demos)

The "real world" the demo sims fly in, copied from dronehacks-sim `assets/` (same values; no link to that repo):

| file | what |
|---|---|
| `course_w0scan2.npz` | the MPPI demo's world and the base of `../main_direction/course_asbuilt.npz` (supplied by the team). Same as `course_w0scan.npz` except wall 0: ~4 cm further east and a 0.84 m wide window at 0.305-1.219 m. |
| `course_w0scan.npz` | Course v3 CAD with wall 0 + the window re-laid where the scan of the built course puts them (dronehacks-sim's "as-built" eval condition; CAD window heights) |
| `course_w0scan_hi.npz` | the same with the window sill / top raised to 0.50 / 1.50 m (the as-built window height is uncertain) |

`<scene>_groundtruth.{bin,json}` are the MPPI demo's planning maps, exported per scene by
`scripts/export_groundtruth_map.py --scene <npz>` (the Rerun MPPI demo rebuilds one when it is missing or older than its scene).

`course_w0scan.npz` / `course_w0scan_hi.npz` are the earlier dronehacks-sim variants, kept for reference (`--scene`).

Format: dronehacks-sim `CourseMesh` npz, NED metres (origin floor centre, z down); load it with
`scripts/real_world_scene.py` (NumPy only). It replaces the `.ply` scan as the source of simulated depth,
`/spoof/scan` and the opaque real-world meshes in the Rerun demos.

The pre-baked CAD prior (`../cad_prior_esdf.bin`, `../arena_prior.*`, `../main_direction/dronehacks-course-v3-obj.zip`) is
intentionally different and unchanged: the demos show the stack generalizing from that prior to this world.

The iSAM2 and full demos fly in `../main_direction/course_asbuilt.npz`: the as-built course the RL policy trains on
(dronehacks-sim's `ftw0_s330` course mods applied at their midpoints to `course_w0scan2.npz`: walls moved / rotated /
added, window 0.45-1.375 m, 0.15 m gap under the walls, launch mark 0.38 m north, as-built checkpoints). Rebuild it with
`PYTHONNOUSERSITE=1 python3 scripts/make_asbuilt_course.py`; `course_asbuilt.json` holds the recipe, launch mark and gates.
Depth in both demos is synthesized from the mesh (`scripts/mesh_depth.py`: exact ray casting + dronehacks-sim's OAK-D
stereo model), not from the `.ply`.
