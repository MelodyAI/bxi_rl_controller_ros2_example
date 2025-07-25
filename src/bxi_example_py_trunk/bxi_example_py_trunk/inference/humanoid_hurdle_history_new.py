# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import os.path as osp
import numpy as np
from typing import Dict
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.inference.exp_filter import expFilter
import onnxruntime as ort


class humanoid_hurdle_onnx_Agent(baseAgent):
    def __init__(self, policy_path):
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
        self.jump = False
        self.agent_count = 0
        self.bootstrap()


    def bootstrap(self):
        "预热用"
        obs_group={
            "dof_pos":np.zeros(23),
            "dof_vel":np.zeros(23),
            "angular_velocity":np.zeros(3),
            "projected_gravity":np.zeros(3),
            "norm_time":np.zeros(1),
            "difficulty":np.zeros(1), 
        }
        self.inference(obs_group)

    def build_observations(self, obs_group):
        obs_dof_pos = obs_group["dof_pos"] * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        motion_time_norm = self.agent_count * 0.02 * 30 / 167
        if motion_time_norm > 0.9:
            motion_time_norm = 0
            self.agent_count = 0
            self.jump =  False
        print(motion_time_norm)

        infer_dt = np.ones((1,))
        difficulty = obs_group["difficulty"]
        # import ipdb; ipdb.set_trace()

        # 本体感知proprioception 9+69+162=240
        prop_obs = np.concatenate((
            obs_base_ang_vel, # 3
            obs_projected_gravity, # 3
            obs_dof_pos, # 23
            obs_dof_vel, # 23
            obs_last_actions, # 23
            np.array([motion_time_norm]), # 1
            infer_dt.flatten(), # 1
            difficulty, # 1
        ),axis=-1)

        if self.is_reset:
            self.prop_obs_history[:] = prop_obs  # 填充所有行
            self.is_reset = False  # 重置标志

        self.prop_obs_history=np.roll(self.prop_obs_history,shift=-1,axis=0)
        self.prop_obs_history[-1,:] = prop_obs

        return self.prop_obs_history
    
    def inference(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

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
        if self.jump:
            self.agent_count += 1

        return dof_pos_target_urdf
    
    def reset(self):
        self.prop_obs_history = np.zeros((self.include_history_steps, self.num_prop_obs_input))
        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter.reset()
        self.is_reset = True  # 标志刚刚重置
        self.agent_count = 0

if __name__=="__main__":
    a=humanoid_hurdle_onnx_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/20250725_140613_elf2_dof23_noOdometry_0_adamimic_stage1.onnx")
    obs_group={
        "dof_pos":np.zeros(23),
        "dof_vel":np.zeros(23),
        "angular_velocity":np.zeros(3),
        "projected_gravity":np.zeros(3),
        "norm_time":np.zeros(1),
        "difficulty":np.zeros(1), 
    }
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    for i in range(100):
        print(a.inference(obs_group))
