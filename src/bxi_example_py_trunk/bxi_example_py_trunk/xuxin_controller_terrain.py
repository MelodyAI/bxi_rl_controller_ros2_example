# Copyright (c) 2025 上海半醒科技有限公司. 保留所有权利. 未经许可，禁止复制、修改或分发
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg 
import sensor_msgs.msg
from threading import Lock
import numpy as np
import time
import sys
import math
from std_msgs.msg import Header,Float32MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
# from bxi_example_py_trunk.inference.humanoid_dh_long import humanoid_dh_long_Agent
# from bxi_example_py_trunk.inference.humanoid_dh_long_onnx import humanoid_dh_long_onnx_Agent
# from bxi_example_py_trunk.inference.humanoid_hurdle import humanoid_hurdle_onnx_Agent
# from bxi_example_py_trunk.inference.humanoid_hurdle_history import humanoid_hurdle_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_hurdle_history_new import humanoid_hurdle_onnx_Agent


import onnxruntime as ort

import termios
import select

robot_name = "elf25"

dof_num = 25
dof_use = 12

# ankle_y_offset = 0.04

joint_name = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    
    "l_hip_z_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_y_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴

    "r_hip_z_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_y_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴

    "l_shld_y_joint",   # 左臂_肩关节_y轴
    "l_shld_x_joint",   # 左臂_肩关节_x轴
    "l_shld_z_joint",   # 左臂_肩关节_z轴
    "l_elb_y_joint",   # 左臂_肘关节_y轴
    "l_elb_z_joint",   # 左臂_肘关节_y轴
    
    "r_shld_y_joint",   # 右臂_肩关节_y轴   
    "r_shld_x_joint",   # 右臂_肩关节_x轴
    "r_shld_z_joint",   # 右臂_肩关节_z轴
    "r_elb_y_joint",    # 右臂_肘关节_y轴
    "r_elb_z_joint",    # 右臂_肘关节_y轴
    )   

joint_nominal_pos = np.array([   # 指定的固定关节角度
    0.0, 0.0, 0.0,
    0,0.0,-0.3,0.6,-0.3,0.0,
    0,0.0,-0.3,0.6,-0.3,0.0,
    0.1,0.0,0.0,-0.3,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    0.1,0.0,0.0,-0.3,0.0],    # 右臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
    dtype=np.float32)

joint_kp = np.array([     # 指定关节的kp，和joint_name顺序一一对应
    500,500,150,
    150,150,150,300,40,40,
    150,150,150,300,40,40,
    2,2,1,2,1,
    2,2,1,2,1,
], dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    5,5,2,
    2,2,2,4,2,2,
    2,2,2,4,2,2,
    0.5,0.5,0.5,0.5,0.5,
    0.5,0.5,0.5,0.5,0.5,
], dtype=np.float32)


isaac_joint_names = [ # isaacgym顺序
    # 0:5
    "l_shld_y_joint",   # 左臂_肩关节_y轴
    "l_shld_x_joint",   # 左臂_肩关节_x轴
    "l_shld_z_joint",   # 左臂_肩关节_z轴
    "l_elb_y_joint",   # 左臂_肘关节_y轴
    "l_elb_z_joint",   # 左臂_肘关节_y轴
    # 5:10
    "r_shld_y_joint",   # 右臂_肩关节_y轴   
    "r_shld_x_joint",   # 右臂_肩关节_x轴
    "r_shld_z_joint",   # 右臂_肩关节_z轴
    "r_elb_y_joint",    # 右臂_肘关节_y轴
    "r_elb_z_joint",    # 右臂_肘关节_y轴
    # 10:11
    "waist_z_joint",
    # 11:17
    "l_hip_z_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_y_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴
    # 17:23
    "r_hip_z_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_y_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴
]
index_isaac_in_mujoco = [joint_name.index(name) for name in isaac_joint_names]


def quat_rotate_inverse(q, v):
    "x,y,z,w"
    q_w = q[-1]
    q_vec = q[:3]
    
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    
    return a - b + c

class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)

        # 策略文件在policy目录下
        self.declare_parameter('/policy_file', 'default_value')
        self.policy_file = self.get_parameter('/policy_file').get_parameter_value().string_value
        print('policy_file:', self.policy_file)

        self.declare_parameter('/policy_file_onnx', 'default_value')
        self.policy_file_onnx = self.get_parameter('/policy_file_onnx').get_parameter_value().string_value
        print('policy_file_onnx:', self.policy_file_onnx)

        # print(index_isaac_in_mujoco)
        self.num_actions = len(index_isaac_in_mujoco)

        qos = QoSProfile(depth=1, durability=qos_profile_sensor_data.durability, reliability=qos_profile_sensor_data.reliability)
        
        self.act_pub = self.create_publisher(bxiMsg.ActuatorCmds, self.topic_prefix+'actuators_cmds', qos)  # CHANGE
        
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, self.topic_prefix+'odom', self.odom_callback, qos)
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.touch_sub = self.create_subscription(bxiMsg.TouchSensor, self.topic_prefix+'touch_sensor', self.touch_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)
        self.height_map_sub = self.create_subscription(Float32MultiArray,'/extracted_elevation_matrix',self.height_map_callback,qos)

        self.rest_srv = self.create_client(bxiSrv.RobotReset, self.topic_prefix+'robot_reset')
        self.sim_rest_srv = self.create_client(bxiSrv.SimulationReset, self.topic_prefix+'sim_reset')
        
        self.timer_callback_group_1 = MutuallyExclusiveCallbackGroup()

        self.lock_in = Lock()
        self.lock_ou = self.lock_in #Lock()
        self.qpos = np.zeros(self.num_actions,dtype=np.double)
        self.qvel = np.zeros(self.num_actions,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)
        
        self.agent = humanoid_hurdle_onnx_Agent(self.policy_file_onnx)
        
        self.vx = 0.1
        self.vy = 0.
        self.dyaw = 0.
        self.height_map = 1.05 - np.zeros(18*9,dtype=np.double) # base_pos_z - height

        self.step = 0
        self.loop_count = 0
        self.dt = 0.02  # loop @100Hz # TODO:
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        # obstacle play
        total_play_time = 0.9 # 0.9m高程图长度 1m/s速度
        self.total_play_count = total_play_time/self.dt
        self.obstacle_height = 0.3
        self.play_count = self.total_play_count + 999
        self.prev_jump_btn = False
    
        self.target_yaw = 0

    def obstacle_play_command_callback(self):
        print("start play obstacle")
        self.play_count = 0

    def obstacle_play_update(self):
        "播放一个20cm障碍的高程图"
        if self.play_count < self.total_play_count:
            percentage = self.play_count/self.total_play_count
            print(f"obstacle percentage:{percentage}")
            
            blank_map = np.zeros((18,9),dtype=np.double)
            start_index = int(18 - percentage * 18)
            start_index = min(start_index,17)
            start_index = max(start_index,0)
            end_index = int(start_index + 4)
            end_index = min(18,end_index)
            end_index = max(0,end_index)
            
            # 20[0:4]cm障碍的高程图 地图90cm[0:18]
            blank_map[start_index:end_index,:] = self.obstacle_height
            self.height_map = (1.05 - blank_map).flatten()
            self.play_count += 1
        else:
            blank_map = np.zeros((18,9),dtype=np.double)
            self.height_map = (1.05 - blank_map).flatten()

    def timer_callback(self):
        
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_rest(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (20./self.dt): # 延迟10s
            self.robot_rest(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(2./self.dt) # 1秒关节缓启动
            if soft_start > 1:
                soft_start = 1
                
            soft_joint_kp = joint_kp * soft_start
                
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = joint_nominal_pos.tolist()
            # msg.pos = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = soft_joint_kp.tolist()
            msg.kd = joint_kd.tolist()
            self.act_pub.publish(msg)
        elif self.step == 2:
            with self.lock_in:
                q = self.qpos
                dq = self.qvel
                quat = self.quat
                omega = self.omega
                
                x_vel_cmd = self.vx
                y_vel_cmd = self.vy
                yaw_vel_cmd = self.dyaw
                height_map = self.height_map

            self.obstacle_play_update()
                                           
            dof_pos = q
            dof_vel = dq
            quat = self.quat
            g_vec = np.array([0.,0.,-1.])
            p_g_vec = quat_rotate_inverse(quat,g_vec)

            x, y, z, w = quat
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            self.base_yaw = np.arctan2(t3, t4)

            ang_scale = 2.
            lin_scale = 2.
            self.dyaw *= ang_scale
            x_vel_cmd *= lin_scale
            if abs(self.dyaw) < 0.1: self.dyaw = 0 # 死区
            self.target_yaw += self.dyaw * self.dt
            yaw_delta = (self.target_yaw - self.base_yaw + np.pi) % (2*np.pi) - np.pi
            # print(x_vel_cmd)
            # print(self.target_yaw, self.base_yaw, yaw_delta)
            difficulty = np.array([1.0])
            obs_group={
                "dof_pos":dof_pos,
                "dof_vel":dof_vel,
                "angular_velocity":omega,
                "commands":np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd]),
                "projected_gravity":p_g_vec,
                "height_map":height_map,
                "yaw_delta":np.array([yaw_delta,yaw_delta]),
                "difficulty": difficulty,
            }

            target_q = self.agent.inference(obs_group)
            
            qpos = joint_nominal_pos.copy()
            qpos[[index_isaac_in_mujoco]] = target_q
            # qpos[4+3] += ankle_y_offset
            # qpos[10+3] += ankle_y_offset
            
            msg = bxiMsg.ActuatorCmds()
            msg.header.frame_id = robot_name
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.actuators_name = joint_name
            msg.pos = qpos.tolist()
            msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
            msg.kp = joint_kp.tolist()
            msg.kd = joint_kd.tolist()
            self.act_pub.publish(msg)

        self.loop_count += 1
    
    def robot_rest(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name
    
        while not self.rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.rest_srv.call_async(req)
        
    def sim_robot_rest(self):        
        req = bxiSrv.SimulationReset.Request()
        req.header.frame_id = robot_name

        base_pose = Pose()
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 1.0
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = 0.0
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 1.0        

        joint_state = JointState()
        joint_state.name = joint_name
        joint_state.position = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.velocity = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.effort = np.zeros(dof_num, dtype=np.float32).tolist()
        
        req.base_pose = base_pose
        req.joint_state = joint_state
    
        while not self.sim_rest_srv.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
            
        self.sim_rest_srv.call_async(req)
    
    def joint_callback(self, msg):
        joint_pos = msg.position
        joint_vel = msg.velocity
        joint_tor = msg.effort
        
        with self.lock_in:
            self.qpos = np.array(joint_pos)[index_isaac_in_mujoco]
            self.qvel = np.array(joint_vel)[index_isaac_in_mujoco]
            # self.qpos[4] -= ankle_y_offset
            # self.qpos[10] -= ankle_y_offset

    def joy_callback(self, msg):
        with self.lock_in:
            self.vx = msg.vel_des.x
            self.vy = msg.vel_des.y
            self.dyaw = msg.yawdot_des

            jump_btn = msg.btn_4 # BT4 = Y按钮控制发送障碍高程图
 
            jump_btn_changed = (jump_btn != self.prev_jump_btn)
            if jump_btn_changed:
                print("jump")
                if self.play_count < self.total_play_count:
                    # 正在发送中 不重复发送
                    pass
                else:
                    self.obstacle_play_command_callback()
                    self.agent.jump = True
                    
            self.prev_jump_btn = jump_btn

    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)

        with self.lock_in:
            self.quat = quat_tmp1
            self.omega = np.array([avel.x, avel.y, avel.z])

    def touch_callback(self, msg):
        foot_force = msg.value
        
    def odom_callback(self, msg): # 全局里程计（上帝视角，仅限仿真使用）
        base_pose = msg.pose
        base_twist = msg.twist

    def height_map_callback(self, msg):
        """处理高程图数据"""
        try:
            # 将一维数组转换为18×9的高程图
            height_map_data = np.array(msg.data, dtype=np.double)
            h = height_map_data.reshape(25,9)
            h = h[:18]
            h = np.flip(h, axis=[0,1])
            h = h.flatten()
            h = - h - 0.228
            np.set_printoptions(formatter={'float': '{:.2f}'.format})
            print(h)

            with self.lock_in:
                self.height_map = h

        except Exception as e:
            self.get_logger().error(f"处理高程图数据失败: {str(e)}")

def main(args=None):
   
    time.sleep(5)
    
    rclpy.init(args=args)
    node = BxiExample()
    
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()

