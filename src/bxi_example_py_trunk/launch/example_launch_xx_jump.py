import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    xml_file_name = "model/xml/elf2-ankle/elf2_ankle_dof25.xml"
    xml_file = os.path.join(get_package_share_path("description"), xml_file_name)

    policy_file_name = "policy/model_xx.jit"
    policy_file = os.path.join(get_package_share_path("bxi_example_py_trunk"), policy_file_name)

    # policy_file_onnx_name = "policy/20250725_140613_elf2_dof23_noOdometry_0_adamimic_stage1.onnx"
    # policy_file_onnx_name = "policy/20250729_044740_elf2_dof23_noOdometry_0_adamimic_stage1_single.onnx"
    # policy_file_onnx_name = "policy/20250729_044646_elf2_dof23_noOdometry_0_adamimic_stage1_single.onnx"
    # policy_file_onnx_name = "policy/20250729_044024_elf2_dof23_noOdometry_0_adamimic_stage1_double.onnx"

    # policy_file_onnx_name = "policy/20250729_044109_elf2_dof23_noOdometry_0_adamimic_stage1_double.onnx"
    # policy_file_onnx_name = "policy、20250803_150053_elf2_dof23_noOdometry_0_adamimic_farjump.onnx"
    # policy_file_onnx_name = "policy/20250803_152657_elf2_dof23_noOdometry_1_adamimic_stage1_highjump.onnx"
    policy_file_onnx_name = "policy/0804_highjump.onnx"
    # policy_file_onnx_name = "policy/0804_dance.onnx"
    
    policy_file_onnx = os.path.join(get_package_share_path("bxi_example_py_trunk"), policy_file_onnx_name)

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
                    {"/policy_file": policy_file},
                    {"/policy_file_onnx": policy_file_onnx},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
