# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import os.path as osp
import numpy as np
from typing import Dict
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.utils.exp_filter import expFilter
import onnxruntime as ort


class humanoid_motion_tracking_Agent(baseAgent):
    def __init__(self, policy_path, motion_time_increment, motion_difficulty, motion_range=[0., 1.]):
        """
        motion_time_increment: 每次step motion_norm_time的增加量
        motion_difficulty: 动作难度
        motion_end: 结束时间
        """
        self.num_actions = 24
        self.num_prop_obs_input = 79
        self.include_history_steps = 5

        providers = [
            'CUDAExecutionProvider',  # 优先使用GPU
            'CPUExecutionProvider'    # 回退到CPU
        ] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 创建推理会话
        self.onnx_session = ort.InferenceSession(
            policy_path,
            providers=providers,
            sess_options=options
        )

        print([ipt.name for ipt in self.onnx_session.get_inputs()])
        self.input_names = [ipt.name for ipt in self.onnx_session.get_inputs()]
        
        self.default_dof_pos = [
            0.,0.,0.,-1.57,0.,
            0.,0.,0.,-1.57,0.,
            0.,
            0.,0.,-0.5,1.0,-0.5,0.,
            0.,0.,-0.5,1.0,-0.5,0.,
        ]
        self.default_dof_pos = np.array(self.default_dof_pos)

        self.obs_scale={
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "ang_vel": 0.25,
            "height_measurements": 5.0,
        }
        self.action_scale = 0.25
        self.prop_obs_history = np.zeros((self.include_history_steps, self.num_prop_obs_input))
        self.clip_observation = 100.
        self.clip_action = 100.

        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter = expFilter(0.6)
        self.is_reset = True
        self.motion_playing = False
        self.agent_count = 0

        self.motion_time_increment = motion_time_increment
        self.motion_difficulty = motion_difficulty
        self.motion_start = motion_range[0]
        self.motion_end = motion_range[1]

        self.bootstrap()

    def bootstrap(self):
        "预热用"
        obs_group={
            "dof_pos":np.zeros(23),
            "dof_vel":np.zeros(23),
            "angular_velocity":np.zeros(3),
            "projected_gravity":np.zeros(3),
        }
        self.inference(obs_group)

    def build_observations(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        obs_dof_pos = obs_group["dof_pos"] * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        motion_time_norm = self.agent_count * self.motion_time_increment
        if motion_time_norm > self.motion_end:
            motion_time_norm = self.motion_end
            self.motion_playing =  False
        print(f"motion percent:{motion_time_norm:.2f}")
        motion_time_norm = np.array([motion_time_norm])

        infer_dt = np.array([0.])
        difficulty = np.array([self.motion_difficulty])
        # import ipdb; ipdb.set_trace()

        # 本体感知proprioception 9+69+162=240
        prop_obs = np.concatenate((
            obs_base_ang_vel, # 3
            obs_projected_gravity, # 3
            obs_dof_pos, # 23
            obs_dof_vel, # 23
            obs_last_actions, # 23
            motion_time_norm, # 1
            infer_dt, # 1
            difficulty, # 1
        ),axis=-1)

        if self.is_reset:
            # self.prop_obs_history[:] = prop_obs  # 填充所有行
            self.is_reset = False  # 重置标志

        self.prop_obs_history=np.roll(self.prop_obs_history,shift=-1,axis=0)
        self.prop_obs_history[-1,:] = prop_obs

        return self.prop_obs_history
    
    def inference(self, obs_group):
        # import ipdb; ipdb.set_trace()
        history_obs = self.build_observations(obs_group).copy()

        input_feed = {
            self.input_names[0]: history_obs.flatten()[None,:].astype(np.float32),
        }
        actions = np.squeeze(self.onnx_session.run(["output"], input_feed)) # test
        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions[:-1] * self.action_scale + self.default_dof_pos

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        if self.motion_playing:
            self.agent_count += 1

        return dof_pos_target_urdf
    
    def reset(self):
        self.prop_obs_history = np.zeros((self.include_history_steps, self.num_prop_obs_input))
        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter.reset()
        self.is_reset = True  # 标志刚刚重置
        self.agent_count = self.motion_start / self.motion_time_increment

if __name__=="__main__":
    motion_agent_dt = 0.02
    video_fps = 30

    # 跳远
    video_buffer_length = 165
    motion_difficulty = 0.65
    motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
    a=humanoid_motion_tracking_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/0805_farjump.onnx",
                                     motion_time_increment, motion_difficulty, motion_end=1.0)

    # 跳高
    video_buffer_length = 108
    # motion_difficulty = 0.3
    motion_difficulty = 0.15
    motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
    b=humanoid_motion_tracking_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/0805_highjump.onnx",
                                     motion_time_increment, motion_difficulty, motion_end=0.6)

    # 跳舞
    video_buffer_length = 588
    motion_difficulty = 0.55
    motion_time_increment = motion_agent_dt * video_fps / video_buffer_length
    c=humanoid_motion_tracking_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/0805_dance.onnx",
                                     motion_time_increment, motion_difficulty, motion_end=1.0)

    obs_group={
        "dof_pos":np.zeros(23),
        "dof_vel":np.zeros(23),
        "angular_velocity":np.zeros(3),
        "projected_gravity":np.zeros(3),
    }
    a.motion_playing = True
    b.motion_playing = True
    c.motion_playing = True
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    for i in range(100):
        print(a.inference(obs_group))
        print(b.inference(obs_group))
        print(c.inference(obs_group))
