# Hardware (what is known, and what training assumes)

| item | value | source |
|---|---|---|
| flight controller | Pixhawk 6C, PX4 v1.17 (`*_v1` topics) | config/versions.yaml |
| companion | Jetson Orin Nano | |
| depth camera | OAK-D S2: passive stereo (no IR projector), 640x400 mono pair, fx ~421, baseline 7.49 cm, 0.35 m min range | organizers' calib |
| rangefinder | VL53L1X | |
| battery | 4S 5000 mAh LiPo (sag model in `sitl/battery.py`; typical constants, not measured) | team |
| motors / props | TBD: thrust table (thrust and current vs throttle at 4S) needed | team |
| arena | 6.10 x 6.10 m net, 3.05 m high; frame of 3.8 cm bars, 5 posts per side, rails at the floor, middle and top | team |

**Training plant** (dronesim `dronehacks/quad.py`):
- Generic 1.5 kg quad: 0.13 m arm, 0.1-10 N per rotor, a 25-35 ms first-order motor lag.
- Lee velocity controller.
- No throttle -> thrust curve: thrust is commanded in newtons directly. A measured motor/prop curve plus
  PX4's throttle linearisation is the next plant upgrade.
