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
joint_nominal_pos = [   # 指定的固定关节角度
    0.,0.,0.,-1.57,0.,
    0.,0.,0.,-1.57,0.,
    0.,
    0.,0.,-0.5,1.0,-0.5,0.,
    0.,0.,-0.5,1.0,-0.5,0.,
]

joint_kp = [ # 指定关节的kp，和joint_name顺序一一对应
    20,20,20,20,20,
    20,20,20,20,20,
    150,
    150,150,150,300,40,40,
    150,150,150,300,40,40,
]
joint_kd = [ # 指定关节的kd，和joint_name顺序一一对应
    1.5,1.5,0.8,1.5,0.8,
    1.5,1.5,0.8,1.5,0.8,
    2,
    2,2,2,4,2,2,
    2,2,2,4,2,2,
]

default_joint_pos = deepcopy(joint_nominal_pos) # = target angles [rad] when action = 0.0

default_joint_pos_dict = {joint_names[index]:joint_nominal_pos[index] for index in range(len(joint_names))}
stiffness_dict = {joint_names[index]:joint_kp[index] for index in range(len(joint_names))}
damping_dict = {joint_names[index]:joint_kd[index] for index in range(len(joint_names))}

high_jump_ref_pos = np.array([
# 0805    
    # -0.2715,  1.3049,  0.3673, -1.1152, -0.0166, -0.6082, -1.5182, -0.6195,
    # -1.5827,  0.0400, -0.0227,  0.2649,  0.1585, -0.2255,  0.3595, -0.2624,
    # -0.1200, -0.2958, -0.0848, -0.2284,  0.3525, -0.2307,  0.0876
# 0807
    0.099, 0.639, -0.569, -0.610, 0.058, 0.095, -0.547, 0.454, -0.654,
    -0.115, -0.095, 0.336, 0.173, -0.650, 0.799, -0.360, -0.206,
    -0.111, -0.158, -0.705, 0.782, -0.287, 0.188
]) # 23
far_jump_ref_pos = np.array([
    # 0.232, 0.585, -0.149, -0.998, -0.112, 0.106, -0.591, 0.207, -1.005,
    # 0.116, 0.053, 0.079, 0.142, -0.364, 0.813, -0.429, -0.143, -0.185,
    # -0.128, -0.483, 0.850, -0.354, 0.118
    0.240, 0.582, -0.159, -1.000, 0.039, -0.001, -0.577, 0.226, -0.952,
    -0.005, 0.074, 0.057, 0.175, -0.296, 0.535, -0.307, -0.153, -0.311,
    -0.149, -0.386, 0.578, -0.261, 0.162    
]) # 23
dance_ref_pos = np.array([
    0.220, 0.587, -0.150, -1.004, -0.111, 0.107, -0.594, 0.204, -1.011,
    0.112, 0.057, 0.074, 0.133, -0.372, 0.817, -0.433, -0.137, -0.185,
    -0.130, -0.475, 0.850, -0.368, 0.121
]) # 23

joint_names_upper_body = [ # isaacgym顺序
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
]

joint_kp_upper_body = [
    20,20,20,20,20,
    20,20,20,20,20,
]
joint_kd_upper_body = [
    1.5,1.5,0.8,1.5,0.8,
    1.5,1.5,0.8,1.5,0.8,
]
high_jump_ref_pos_upper_body = np.array([high_jump_ref_pos[joint_names.index(name)] for name in joint_names_upper_body])
far_jump_ref_pos_upper_body = np.array([far_jump_ref_pos[joint_names.index(name)] for name in joint_names_upper_body])
dance_ref_pos_upper_body = np.array([dance_ref_pos[joint_names.index(name)] for name in joint_names_upper_body])

if __name__=="__main__":
    print(high_jump_ref_pos_upper_body)