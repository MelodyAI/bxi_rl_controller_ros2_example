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
import json
from std_msgs.msg import Header,Float32MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
# from bxi_example_py_trunk.inference.humanoid_dh_long import humanoid_dh_long_Agent
# from bxi_example_py_trunk.inference.humanoid_dh_long_onnx import humanoid_dh_long_onnx_Agent
# from bxi_example_py_trunk.inference.humanoid_hurdle import humanoid_hurdle_onnx_Agent
# from bxi_example_py_trunk.inference.humanoid_hurdle_history import humanoid_hurdle_onnx_Agent
# from bxi_example_py_trunk.inference.humanoid_hurdle_history_v2 import humanoid_hurdle_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_hurdle_history_v3 import humanoid_hurdle_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_dh_long_comp_onnx import humanoid_dh_long_comp_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_walk import humanoid_walk_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_walk_stand_height import humanoid_walk_stand_height_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_motion_tracking import humanoid_motion_tracking_Agent

from bxi_example_py_trunk.utils.legged_math import quat_rotate_inverse,quaternion_to_euler_array
from bxi_example_py_trunk.utils.counter import Counter, recoverCounter

import bxi_example_py_trunk.joint_info.trunk_12dof as joint_info_12
import bxi_example_py_trunk.joint_info.trunk_12dof_example as joint_info_12_example
import bxi_example_py_trunk.joint_info.trunk_23dof as joint_info_23
import bxi_example_py_trunk.joint_info.trunk_25dof as joint_info_25

robot_name = "elf25"

dof_num = 25
dof_use = 12

# ankle_y_offset = 0.04
ankle_y_offset = 0.0

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
    500,500,300,
    100,100,100,150,30,10,
    100,100,100,150,30,10,
    20,20,10,20,10,
    20,20,10,20,10], dtype=np.float32)

joint_kd = np.array([  # 指定关节的kd，和joint_name顺序一一对应
    5,5,3,
    2,2,2,2.5,1,1,
    2,2,2,2.5,1,1,
    1,1,0.8,1,0.8,
    1,1,0.8,1,0.8], dtype=np.float32)

torque_limit = np.array([
    20,40,100,120,50,20,
    20,40,100,120,50,20,
    50,50,50,
    27,27,7,27,7,
    27,27,7,27,7,
])

index_isaac_in_mujoco_12 = [joint_name.index(name) for name in joint_info_12.joint_names]
index_isaac_in_mujoco_12_example = [joint_name.index(name) for name in joint_info_12_example.joint_names]
index_isaac_in_mujoco_23 = [joint_name.index(name) for name in joint_info_23.joint_names]
index_isaac_in_mujoco_23_upper_body = [joint_name.index(name) for name in joint_info_23.joint_names_upper_body]

class robotState:
    stand = 1
    stand_to_motion = 2
    motion = 3
    motion_to_stand = 4

class motionType:
    high_jump = 1
    far_jump = 2
    dance = 3

class BxiExample(Node):

    def __init__(self):

        super().__init__('bxi_example_py')
        
        self.declare_parameter('/topic_prefix', 'default_value')
        self.topic_prefix = self.get_parameter('/topic_prefix').get_parameter_value().string_value
        print('topic_prefix:', self.topic_prefix)

        self.declare_parameter('/policy_file_dict', json.dumps({}))
        policy_file_json = self.get_parameter('/policy_file_dict').value
        self.policy_file_dict = json.loads(policy_file_json)
        print('policy_file:')
        for key,value in self.policy_file_dict.items():
            print(key,": ",value)

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
        self.qpos = np.zeros(25,dtype=np.double)
        self.qvel = np.zeros(25,dtype=np.double)
        self.omega = np.zeros(3,dtype=np.double)
        self.quat = np.zeros(4,dtype=np.double)
        
        # self.walk_agent = humanoid_walk_onnx_Agent(self.policy_file_dict["walk_example"])
        self.walk_agent = humanoid_walk_stand_height_onnx_Agent(self.policy_file_dict["walk_example_height"])
        
        motion_agent_dt = 0.02
        video_fps = 30
        # 跳远
        video_buffer_length = 165
        motion_difficulty = 0.15 # [0.15, 0.65]
        motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
        self.far_jump_agent=humanoid_motion_tracking_Agent(self.policy_file_dict["far_jump"],
                                                            motion_time_increment, motion_difficulty)

        # 跳高
        # video_buffer_length = 108 # 0805
        video_buffer_length = 261 # 0807
        motion_difficulty = 0.15 # [0.15, 0.35]
        motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
        self.high_jump_agent=humanoid_motion_tracking_Agent(self.policy_file_dict["high_jump"],
                                                            motion_time_increment, motion_difficulty, motion_range=[0.3,0.65])

        # 跳舞
        video_buffer_length = 588
        motion_difficulty = 0.55
        motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
        self.dance_agent=humanoid_motion_tracking_Agent(self.policy_file_dict["dance"],
                                                        motion_time_increment, motion_difficulty)

        self.vx = 0.
        self.vy = 0.
        self.dyaw = 0.
        self.stand_height = 1.
        self.height_map = 1.05 - np.zeros(18*9,dtype=np.double) # base_pos_z - height

        self.step = 0
        self.loop_count = 0
        self.loop_count_step_2 = 0
        self.loop_dt = 0.01  # loop @100Hz
        self.timer = self.create_timer(self.loop_dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        # obstacle play
        total_play_time = 0.9 # 0.9m高程图长度 1m/s速度
        self.total_play_count = total_play_time/self.loop_dt
        self.obstacle_height = 0.3
        self.play_count = self.total_play_count + 999
        self.btn_8_prev = False
        self.btn_9_prev = False
        self.btn_10_prev = False
    
        self.target_yaw = 0

        self.jump = False
        self.jump_count = 0
        self.stand_to_motion_counter = None
        self.motion_to_stand_counter = None
        self.state = robotState.stand
        self.dance_btn_changed = False
        self.far_jump_btn_changed = False
        self.high_jump_btn_changed = False
        self.motion_type = None

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

    def state_machine(self):
        "状态机"
        if self.state==robotState.stand:
            if self.high_jump_btn_changed:
                # 进入跳跃准备状态
                self.state = robotState.stand_to_motion
                upper_body_current = self.qpos[index_isaac_in_mujoco_23_upper_body]
                upper_body_target = joint_info_23.high_jump_ref_pos_upper_body
                self.stand_to_motion_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
                self.motion_type = motionType.high_jump
                print("state: stand_to_motion [high jump]")
            elif self.far_jump_btn_changed:
                self.state = robotState.stand_to_motion
                upper_body_current = self.qpos[index_isaac_in_mujoco_23_upper_body]
                upper_body_target = joint_info_23.far_jump_ref_pos_upper_body
                self.stand_to_motion_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
                self.motion_type = motionType.far_jump
                print("state: stand_to_motion [far jump]")
            elif self.dance_btn_changed:
                self.state = robotState.stand_to_motion
                upper_body_current = self.qpos[index_isaac_in_mujoco_23_upper_body]
                upper_body_target = joint_info_23.dance_ref_pos_upper_body
                self.stand_to_motion_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
                self.motion_type = motionType.dance
                print("state: stand_to_motion [dance]")
            else:
                pass
            
        elif self.state==robotState.stand_to_motion:
            self.stand_to_motion_counter.step()
            if self.stand_to_motion_counter.finished:
                self.state=robotState.motion
                self.stand_to_motion_counter = None
                if self.motion_type == motionType.high_jump:
                    self.high_jump_agent.reset()
                    self.high_jump_agent.motion_playing = True
                    print("state: motion [jump]")
                elif self.motion_type == motionType.far_jump:
                    self.far_jump_agent.reset()
                    self.far_jump_agent.motion_playing = True
                    print("state: motion [far_jump]")
                elif self.motion_type == motionType.dance:
                    self.dance_agent.reset()
                    self.dance_agent.motion_playing = True
                    print("state: motion [dance]")

        elif self.state==robotState.motion:
            # 判断动作结束
            # if self.motion_type == motionType.high_jump and (not self.high_jump_agent.motion_playing):
            #     self.state=robotState.motion_to_stand
            #     upper_body_current = self.qpos
            #     upper_body_target = joint_nominal_pos
            #     self.motion_to_stand_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
            #     self.walk_agent.reset()
            #     print("state: motion_to_stand")

            # if self.motion_type == motionType.far_jump and (not self.far_jump_agent.motion_playing):
            #     self.state=robotState.motion_to_stand
            #     upper_body_current = self.qpos
            #     upper_body_target = joint_nominal_pos
            #     self.motion_to_stand_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
            #     self.walk_agent.reset()
            #     print("state: motion_to_stand")

            # if self.motion_type == motionType.dance and (not self.dance_agent.motion_playing):
            #     self.state=robotState.motion_to_stand
            #     upper_body_current = self.qpos
            #     upper_body_target = joint_nominal_pos
            #     self.motion_to_stand_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
            #     self.walk_agent.reset()
            #     print("state: motion_to_stand")

            if self.high_jump_btn_changed:
                self.high_jump_btn_changed = False
                # # 进入跳跃准备状态
                # self.state = robotState.stand_to_motion
                # upper_body_current = self.qpos[index_isaac_in_mujoco_23_upper_body]
                # upper_body_target = joint_info_23.high_jump_ref_pos_upper_body
                # self.stand_to_motion_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
                # self.motion_type = motionType.high_jump
                # print("state: stand_to_motion [high jump]")  

                # 直接启动
                self.high_jump_agent.reset()
                self.high_jump_agent.motion_playing = True
                print("state: motion [jump]")       

        elif(self.state==robotState.motion_to_stand):
            self.motion_to_stand_counter.step()
            if self.motion_to_stand_counter.finished:
                self.state=robotState.stand
                self.motion_to_stand_counter = None
                print("state: stand")
  
        else:
            #其他状态机情况
            raise Exception

    def timer_callback(self):
        # ptyhon 与 rclpy 多线程不太友好，这里使用定时间+简易状态机运行a
        if self.step == 0:
            self.robot_rest(1, False) # first reset
            print('robot reset 1!')
            self.step = 1
            return
        elif self.step == 1 and self.loop_count >= (10./self.loop_dt): # 延迟10s
            self.robot_rest(2, True) # first reset
            print('robot reset 2!')
            self.loop_count = 0
            self.step = 2
            return
        
        if self.step == 1:
            soft_start = self.loop_count/(1./self.loop_dt) # 1秒关节缓启动
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
                dof_pos = self.qpos
                dof_vel = self.qvel
                base_quat = self.quat
                base_ang_vel = self.omega
                
                x_vel_cmd = self.vx
                y_vel_cmd = self.vy
                yaw_vel_cmd = self.dyaw
                height_map = self.height_map
            # start = time.time()

            # self.obstacle_play_update()
            self.state_machine()

            gravity_vec = np.array([0.,0.,-1.])
            projected_gravity_vec = quat_rotate_inverse(base_quat, gravity_vec)

            rpy_angle = quaternion_to_euler_array(base_quat)
            rpy_angle[rpy_angle > math.pi] -= 2 * math.pi
            base_yaw = rpy_angle[2]

            ang_scale = 2.
            if abs(self.dyaw) < 0.1: self.dyaw = 0 # 死区
            if self.loop_count_step_2 == 0:
                self.target_yaw = base_yaw # 第一帧传感器yaw设为起点
            else:
                self.target_yaw += self.dyaw * self.loop_dt
            yaw_delta = (self.target_yaw - base_yaw) * ang_scale
            yaw_delta = (yaw_delta + np.pi) % (2*np.pi) - np.pi

            dof_pos_target = None
            joint_kp_send = joint_kp.copy()
            joint_kd_send = joint_kd.copy()
            obs_group={
                "angular_velocity":base_ang_vel,
                "commands":np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd]),
                "projected_gravity":projected_gravity_vec,
                "euler_angle":rpy_angle,
                "height_map":height_map,
                "yaw_delta":np.array([yaw_delta,yaw_delta]),
                "stand_height":np.array([self.stand_height]),
            }
            # dof_pos[7] -= ankle_y_offset
            # dof_pos[13] -= ankle_y_offset
            if ((self.state == robotState.stand)or
                (self.state == robotState.stand_to_motion)or
                (self.state == robotState.motion_to_stand)):
                obs_group["dof_pos"] = dof_pos[index_isaac_in_mujoco_12_example]
                obs_group["dof_vel"] = dof_vel[index_isaac_in_mujoco_12_example]
            elif self.state == robotState.motion:
                obs_group["dof_pos"] = dof_pos[index_isaac_in_mujoco_23]
                obs_group["dof_vel"] = dof_vel[index_isaac_in_mujoco_23]
            else:
                raise Exception

            if self.state == robotState.stand:
                agent_out = self.walk_agent.inference(obs_group)
                dof_pos_target = joint_nominal_pos.copy()
                dof_pos_target[index_isaac_in_mujoco_12_example] = agent_out
                joint_kp_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kp
                joint_kd_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kd

            elif self.state == robotState.stand_to_motion:
                # 进入跳跃状态后 先把手臂抬起来到启动位置
                dof_pos_target = joint_nominal_pos.copy()
                blend = min(self.stand_to_motion_counter.percent * 1.5, 1.0) # 预留一些时间保持姿势
                print("stand_to_motion:",blend)

                # 上半身插值
                dof_pos_target[index_isaac_in_mujoco_23_upper_body] = self.stand_to_motion_counter.get_dof_pos_by_other_percent(blend)
                joint_kp_send[index_isaac_in_mujoco_23_upper_body] = joint_info_23.joint_kp_upper_body
                joint_kd_send[index_isaac_in_mujoco_23_upper_body] = joint_info_23.joint_kd_upper_body

                # 下半身仍然用走路的控制
                agent_out = self.walk_agent.inference(obs_group)
                dof_pos_target[index_isaac_in_mujoco_12_example] = agent_out

                joint_kp_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kp
                joint_kd_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kd

            elif self.state == robotState.motion:
                if (self.loop_count % 2 == 0): # jump agent是50hz
                    # print("jump inference")
                    if self.motion_type==motionType.high_jump:
                        agent_out = self.high_jump_agent.inference(obs_group)
                    elif self.motion_type==motionType.far_jump:
                        agent_out = self.far_jump_agent.inference(obs_group)
                    elif self.motion_type==motionType.dance:
                        agent_out = self.dance_agent.inference(obs_group)

                    dof_pos_target = joint_nominal_pos.copy()
                    dof_pos_target[index_isaac_in_mujoco_23] = agent_out
                    joint_kp_send[index_isaac_in_mujoco_23] = joint_info_23.joint_kp
                    joint_kd_send[index_isaac_in_mujoco_23] = joint_info_23.joint_kd
                else:
                    # 50hz空白的一帧发送上一帧相同指令
                    # print("jump inference skip")
                    dof_pos_target = self.dof_pos_target_last
                    joint_kp_send = self.joint_kp_send_last
                    joint_kd_send = self.joint_kd_send_last

            elif self.state == robotState.motion_to_stand:
                dof_pos_target = joint_nominal_pos.copy()
                blend = max(self.motion_to_stand_counter.percent * 1.5 - 0.5, 0.)
                print("motion_to_stand:", blend)

                # 上半身插值
                dof_pos_target[index_isaac_in_mujoco_23_upper_body] = self.motion_to_stand_counter.get_dof_pos_by_other_percent(blend)[index_isaac_in_mujoco_23_upper_body]

                # 下半身仍然用走路的控制
                agent_out = self.walk_agent.inference(obs_group)
                blend2 = min(self.motion_to_stand_counter.percent * 4.0, 1.0) # 0.5s
                # dof_pos_target[index_isaac_in_mujoco_12_example] = self.motion_to_stand_counter.dof_pos_start[index_isaac_in_mujoco_12_example] * (1-blend2) + agent_out * blend2
                dof_pos_target[index_isaac_in_mujoco_12_example] = agent_out
        
                # joint_kp_send[index_isaac_in_mujoco_12_example] = np.array(joint_info_12_example.joint_kp, dtype=np.float32) * (0.5 + 0.5 * self.motion_to_stand_counter.percent)
                joint_kp_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kp
                joint_kd_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kd
            else:
                raise Exception

            # dof_pos_target[7] += ankle_y_offset
            # dof_pos_target[13] += ankle_y_offset

            # 软限位
            # upper_limit = dof_pos + (torque_limit - dof_vel * joint_kd) / joint_kp
            # lower_limit = dof_pos + (-torque_limit - dof_vel * joint_kd) / joint_kp
            # qpos = qpos.clip(lower_limit, upper_limit)

            # torque = (qpos - dof_pos) * joint_kp + dof_vel * joint_kd
            # print("torque",torque)

            self.send_to_motor(dof_pos_target, joint_kp_send, joint_kd_send)
            self.dof_pos_target_last = dof_pos_target.copy()
            self.joint_kp_send_last = joint_kp_send.copy()
            self.joint_kd_send_last = joint_kd_send.copy()

            # end = time.time()
            # print("calculate time:", end-start)
            self.loop_count_step_2 += 1
        self.loop_count += 1
    
    def send_to_motor(self, dof_pos_target, joint_kp, joint_kd):
        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = joint_name
        msg.pos = dof_pos_target.tolist()
        msg.vel = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.torque = np.zeros(dof_num, dtype=np.float32).tolist()
        msg.kp = joint_kp.tolist()
        msg.kd = joint_kd.tolist()
        self.act_pub.publish(msg)

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
            self.qpos = np.array(joint_pos)
            self.qvel = np.array(joint_vel)
            # self.qpos[4] -= ankle_y_offset
            # self.qpos[10] -= ankle_y_offset

    def joy_callback(self, msg):
        with self.lock_in:
            self.vx = msg.vel_des.x
            self.vy = msg.vel_des.y
            self.dyaw = msg.yawdot_des
            self.stand_height = min(msg.height_des, 3.0)

            btn_8 = msg.btn_8 # A
            # btn_9 = msg.btn_9 # X
            btn_10 = msg.btn_10 # Y
            if self.step < 2:
                self.btn_8_prev = btn_8
                # self.btn_9_prev = btn_9
                self.btn_10_prev = btn_10

            self.dance_btn_changed = (btn_8 != self.btn_8_prev)
            # self.far_jump_btn_changed = (btn_9 != self.btn_9_prev)
            self.high_jump_btn_changed = (btn_10 != self.btn_10_prev)

            self.btn_8_prev = btn_8
            # self.btn_9_prev = btn_9
            self.btn_10_prev = btn_10

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

