"""ament_python package for skybeta's team_ws (clone this repo into ~/team_ws/src/, build with team_build.sh)."""
from glob import glob

from setuptools import setup

package_name = 'policynav'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/skybeta*.launch.py')),
        ('share/' + package_name + '/policies', glob('policies/*.npz')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Colin Balfour',
    maintainer_email='cbalfour@wpi.edu',
    description='Trained navigation policy + stereo depth for the skybeta drone',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'skybeta_policy = policynav.skybeta_node:main',
        'stereo_depth = policynav.depth_node:main',
        'handheld = policynav.handheld_node:main',
    ]},
)
