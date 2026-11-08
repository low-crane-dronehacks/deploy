# Drone parameter dumps for SITL

Put a dump of the real FC here (QGC "Save to file" / MAVProxy `param save`, tab-separated `1 1 NAME VALUE TYPE` lines,
as in skybeta's `px4/params/fleet/*.params`) and start the rig with it:

    PX4_PARAMS_DUMP=sitl/params/drone17.params sitl/run_rig.sh up

`sitl/params.py` takes only the estimator / failsafe / navigation / envelope families from it (`DUMP_ALLOW`), never
the airframe-specific ones (rate gains, rotor geometry, hover thrust, calibration, RC, serial ports): the SITL vehicle
is an x500, so those would make it fly wrong. The airframe the rig generates lists every value and where it came from:
`run_rig.sh sh`, then `cat /px4/build/px4_sitl_default/etc/init.d-posix/airframes/4090_gz_x500_skybeta`.
