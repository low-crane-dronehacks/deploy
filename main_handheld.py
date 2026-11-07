"""Entry point for the handheld / data-collection mode (camera + policy, never commands PX4):

    ros2 launch launch/policynav.launch.py hardware:=handheld        (or python3 main_handheld.py --ros-args ...)
"""
import sys

from policynav import handheld_node

if __name__ == '__main__':
    handheld_node.main(sys.argv)
