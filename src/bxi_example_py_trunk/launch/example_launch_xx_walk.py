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
        # "walk_main": "policy/Aug07_12-29-51_model_522500.onnx",
        "walk_main": "policy/Aug14_18-14-49_model_450000.onnx",
        # "walk_comp": "policy/Aug07_12-29-51_model_522500_compensate.onnx",
        # "walk_comp": "policy/Aug07_12-29-51_model_522500_compensate_damping0d1.onnx",
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
                executable="xuxin_controller_walk",
                name="xuxin_controller_walk",
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
