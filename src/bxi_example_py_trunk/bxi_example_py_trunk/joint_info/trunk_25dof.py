# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
from copy import deepcopy
joint_names = [ # isaacgym顺序
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
    # 10:13
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    # 13:19
    "l_hip_z_joint",   # 左腿_髋关节_z轴
    "l_hip_x_joint",   # 左腿_髋关节_x轴
    "l_hip_y_joint",   # 左腿_髋关节_y轴
    "l_knee_y_joint",   # 左腿_膝关节_y轴
    "l_ankle_y_joint",   # 左腿_踝关节_y轴
    "l_ankle_x_joint",   # 左腿_踝关节_x轴
    # 19:25
    "r_hip_z_joint",   # 右腿_髋关节_z轴    
    "r_hip_x_joint",   # 右腿_髋关节_x轴
    "r_hip_y_joint",   # 右腿_髋关节_y轴
    "r_knee_y_joint",   # 右腿_膝关节_y轴
    "r_ankle_y_joint",   # 右腿_踝关节_y轴
    "r_ankle_x_joint",   # 右腿_踝关节_x轴
]
joint_nominal_pos = [   # 指定的固定关节角度
    0.7,0.2,-0.1,-1.5,0.,
    0.7,-0.2,0.1,-1.5,0.,
    0.,0.,0.,
    0.,0.,-0.3,0.6,-0.3,0.,
    0.,0.,-0.3,0.6,-0.3,0.,
]

joint_kp = [     # 指定关节的kp，和joint_name顺序一一对应
    30,30,30,30,30,
    30,30,30,30,30,
    1000,1000,300,
    100,100,100,100,20,10,
    100,100,100,100,20,10,
]
joint_kd = [  # 指定关节的kd，和joint_name顺序一一对应
    1,1,0.8,1,0.8,
    1,1,0.8,1,0.8,
    10,10,3,
    3,3,3,3,0.5,0.5,
    3,3,3,3,0.5,0.5,
]

default_joint_pos = deepcopy(joint_nominal_pos) # = target angles [rad] when action = 0.0
# 手臂默认竖直放下
stand_joint_pos = [ # 只影响reward_dof_pos计算 可以和default不一样
    0.,0.,0.,0.,0.,
    0.,0.,0.,0.,0.,
    0.,0.,0.,
    0.,0.,-0.3,0.6,-0.3,0.,
    0.,0.,-0.3,0.6,-0.3,0.,
]
default_joint_pos_dict = {joint_names[index]:joint_nominal_pos[index] for index in range(len(joint_names))}
stiffness_dict = {joint_names[index]:joint_kp[index] for index in range(len(joint_names))}
damping_dict = {joint_names[index]:joint_kd[index] for index in range(len(joint_names))}

