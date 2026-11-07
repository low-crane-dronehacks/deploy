"""Is a cuVSLAM frame VISUALLY tracked? PURE (shipped next to cuvslam_lib_node.py in its image).

In Inertial mode the library returns an IMU-propagated pose when visual tracking fails, without a
flag; the node therefore counts a frame as visually tracked only with at least min_obs tracked
features. UNKNOWN evidence fails closed (review 2026-10-17: a failed observation query, n_obs < 0,
used to count as tracked)."""


def visually_tracked(pose_ok, n_obs, min_obs):
    """pose_ok: the library returned a pose; n_obs: tracked features (< 0 = the query failed)."""
    return bool(pose_ok) and n_obs is not None and n_obs >= 0 and n_obs >= int(min_obs)
