import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

def generate_launch_description():

    xml_file_name = "model/xml/elf2-ankle/elf2_ankle_dof25.xml"
    xml_file = os.path.join(get_package_share_path("description"), xml_file_name)

    policy_file_dict = {
        # "high_jump": "policy/0805_highjump.onnx",
        # "high_jump": "policy/0807_highjump.onnx",
        "high_jump": "policy/0809_highjump.onnx",
        "far_jump": "policy/0805_farjump.onnx",
        # "dance": "policy/0805_dance.onnx",
        "dance": "policy/0810_dance.onnx",
        "walk_example": "policy/walk_example.onnx",
        "walk_example_height": "policy/walk_example_height.onnx",
    }
    for key, value in policy_file_dict.items():
        policy_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_trunk"), value)

    return LaunchDescription(
        [
            Node(
                package="mujoco",
                executable="simulation",
                name="simulation_mujoco",
                output="screen",
                parameters=[
                    {"simulation/model_file": xml_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_trunk",
                executable="xuxin_controller_terrain",
                name="xuxin_controller_terrain",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                    {"/policy_file_dict": json.dumps(policy_file_dict)},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
