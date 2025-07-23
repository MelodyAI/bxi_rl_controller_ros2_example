import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    policy_file_name = "policy/model_xx.jit"
    policy_file = os.path.join(get_package_share_path("bxi_example_py_trunk"), policy_file_name)

    policy_file_onnx_name = "policy/20250723_203911_elf2_dof23_0_adamimic_stage1.onnx"
    policy_file_onnx = os.path.join(get_package_share_path("bxi_example_py_trunk"), policy_file_onnx_name)

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
                executable="xuxin_controller_terrain",
                name="xuxin_controller_terrain",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/policy_file": policy_file}, # not used
                    {"/policy_file_onnx": policy_file_onnx},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
