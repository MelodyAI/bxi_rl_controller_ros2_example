import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import json

def generate_launch_description():

    policy_file_dict = {
        "walk_main": "policy/Aug07_12-29-51_model_522500.onnx",
        "walk_comp": "policy/Aug07_12-29-51_model_522500_compensate.onnx",
    }
    for key, value in policy_file_dict.items():
        policy_file_dict[key] = os.path.join(get_package_share_path("bxi_example_py_trunk"), value)

    return LaunchDescription(
        [
            Node(
                package="hardware_ankle",
                executable="hardware_ankle",
                name="hardware_ankle",
                output="screen",
                parameters=[
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_trunk",
                executable="xuxin_controller_walk",
                name="xuxin_controller_walk",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/policy_file_dict": json.dumps(policy_file_dict)},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
