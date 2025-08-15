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
from copy import deepcopy
from std_msgs.msg import Header,Float32MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from bxi_example_py_trunk.inference.humanoid_walk_stand_height import humanoid_walk_stand_height_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_motion_tracking import humanoid_motion_tracking_Agent

from bxi_example_py_trunk.utils.legged_math import quat_rotate_inverse,quaternion_to_euler_array
from bxi_example_py_trunk.utils.counter import recoverCounter

import bxi_example_py_trunk.joint_info.trunk_12dof as joint_info_12
import bxi_example_py_trunk.joint_info.trunk_12dof_example as joint_info_12_example
import bxi_example_py_trunk.joint_info.trunk_23dof as joint_info_23
import bxi_example_py_trunk.joint_info.trunk_25dof as joint_info_25

robot_name = "elf25"

dof_num = 25
dof_use = 12

ankle_y_offset = -0.0 # +向后倒 -向前倒

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
    50,50,50,
    20,40,100,120,50,20,
    20,40,100,120,50,20,
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
        
        self.joint_sub = self.create_subscription(sensor_msgs.msg.JointState, self.topic_prefix+'joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(sensor_msgs.msg.Imu, self.topic_prefix+'imu_data', self.imu_callback, qos)
        self.joy_sub = self.create_subscription(bxiMsg.MotionCommands, 'motion_commands', self.joy_callback, qos)

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

        # 跳高
        # video_buffer_length = 108 # 0805
        video_buffer_length = 261 # 0807
        motion_difficulty = 0.15 # [0.15, 0.35]
        motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
        self.high_jump_agent=humanoid_motion_tracking_Agent(self.policy_file_dict["high_jump"],
                                                            motion_time_increment, motion_difficulty, motion_range=[0.3,0.65])

        self.vx = 0.
        self.vy = 0.
        self.dyaw = 0.
        self.stand_height = 1.

        self.step = 0
        self.loop_count = 0
        self.loop_count_step_2 = 0
        self.loop_dt = 0.01  # loop @100Hz
        self.timer = self.create_timer(self.loop_dt, self.timer_callback, callback_group=self.timer_callback_group_1)

        self.high_jump_btn_prev = False
        self.high_jump_btn_changed = False
        self.stop_btn_prev = False
        self.stop_btn_changed = False

        self.target_yaw = 0

        self.stand_to_motion_counter = None
        self.motion_to_stand_counter = None
        self.state = robotState.stand
        self.motion_type = None

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
            
        elif self.state==robotState.stand_to_motion:
            self.stand_to_motion_counter.step()
            if self.stand_to_motion_counter.finished:
                self.state=robotState.motion
                self.stand_to_motion_counter = None
                if self.motion_type == motionType.high_jump:
                    self.high_jump_agent.reset()
                    self.high_jump_agent.motion_playing = True # 进入蹲的姿势以后等待按键再跳
                    print("state: motion [high jump]")

        elif self.state==robotState.motion:
            if self.motion_type == motionType.high_jump:
                if self.high_jump_btn_changed:
                    # 结束了之后不返回站立 连续跳
                    self.high_jump_btn_changed = False
                    self.high_jump_agent.reset() # 不reset好一点
                    self.high_jump_agent.motion_playing = True
                    print("state: motion [jump]")
                if self.stop_btn_changed:
                    # 手动退出
                    self.stop_btn_changed = False
                    self.state=robotState.motion_to_stand
                    upper_body_current = self.qpos
                    upper_body_target = joint_nominal_pos
                    self.motion_to_stand_counter = recoverCounter(2.0/self.loop_dt, upper_body_current, upper_body_target)
                    self.walk_agent.reset()
                    print("state: motion_to_stand")

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
            # start = time.time()

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
                "yaw_delta":np.array([yaw_delta,yaw_delta]),
                "stand_height":np.array([self.stand_height]),
            }
            dof_pos[7] -= ankle_y_offset
            dof_pos[13] -= ankle_y_offset
            obs_group_12 = {
                "dof_pos": dof_pos[index_isaac_in_mujoco_12_example],
                "dof_vel": dof_vel[index_isaac_in_mujoco_12_example],
                **obs_group,
            }
            obs_group_23 = {
                "dof_pos": dof_pos[index_isaac_in_mujoco_23],
                "dof_vel": dof_vel[index_isaac_in_mujoco_23],
                **obs_group,
            }

            if self.state == robotState.stand:
                agent_out = self.walk_agent.inference(obs_group_12)
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
                agent_out = self.walk_agent.inference(obs_group_12)
                dof_pos_target[index_isaac_in_mujoco_12_example] = agent_out

                joint_kp_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kp
                joint_kd_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kd

            elif self.state == robotState.motion:
                if (self.loop_count % 2 == 0): # jump agent是50hz
                    # print("jump inference")
                    if self.motion_type==motionType.high_jump:
                        agent_out = self.high_jump_agent.inference(obs_group_23)

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
                if (self.loop_count % 2 == 0): # jump agent是50hz
                    # print("jump inference")
                    if self.motion_type==motionType.high_jump:
                        agent_out = self.high_jump_agent.inference(obs_group_23)

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
                dof_pos_target_jump = dof_pos_target.copy()
                joint_kp_send_jump = joint_kp_send.copy()
                joint_kd_send_jump = joint_kd_send.copy()

                dof_pos_target = joint_nominal_pos.copy()
                joint_kp_send = joint_kp.copy()
                joint_kd_send = joint_kd.copy()
                agent_out = self.walk_agent.inference(obs_group_12)
                dof_pos_target[index_isaac_in_mujoco_12_example] = agent_out
                joint_kp_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kp
                joint_kd_send[index_isaac_in_mujoco_12_example] = joint_info_12_example.joint_kd
                dof_pos_target_walk = dof_pos_target.copy()
                joint_kp_send_walk = joint_kp_send.copy()
                joint_kd_send_walk = joint_kd_send.copy()

                blend = self.motion_to_stand_counter.percent
                print("blend:",blend)
                dof_pos_target = dof_pos_target_jump * (1-blend) + dof_pos_target_walk * blend
                joint_kp_send = joint_kp_send_jump * (1-blend) + joint_kp_send_walk * blend
                joint_kd_send = joint_kd_send_jump * (1-blend) + joint_kd_send_walk * blend

            else:
                raise Exception

            dof_pos_target[7] += ankle_y_offset
            dof_pos_target[13] += ankle_y_offset

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
            stand_height = msg.height_des
            stand_height = min(stand_height, 3.0)
            stand_height = max(stand_height, 1.0)
            self.stand_height = stand_height

            high_jump_btn = msg.btn_7 # Y
            stop_btn = msg.btn_10 # B
            if self.step < 2:
                self.high_jump_btn_prev = high_jump_btn
                self.stop_btn_prev = stop_btn

            self.high_jump_btn_changed = (high_jump_btn != self.high_jump_btn_prev)
            self.stop_btn_changed = (stop_btn != self.stop_btn_prev)

            self.high_jump_btn_prev = high_jump_btn
            self.stop_btn_prev = stop_btn

    def imu_callback(self, msg):
        quat = msg.orientation
        avel = msg.angular_velocity
        acc = msg.linear_acceleration

        quat_tmp1 = np.array([quat.x, quat.y, quat.z, quat.w]).astype(np.double)

        with self.lock_in:
            self.quat = quat_tmp1
            self.omega = np.array([avel.x, avel.y, avel.z])


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

