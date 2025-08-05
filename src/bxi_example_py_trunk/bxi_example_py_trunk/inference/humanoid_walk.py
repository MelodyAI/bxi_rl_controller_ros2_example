import numpy as np
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.utils.exp_filter import expFilter
import time
import onnxruntime as ort
import math

class humanoid_walk_onnx_Agent(baseAgent):
    def __init__(self, policy_path):
        self.num_prop_obs_input = 47
        self.long_history = 66
        self.num_actions = 12

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
        
        self.default_dof_pos = [0., 0., -0.3, 0.6, -0.3, 0.,
                                0., 0., -0.3, 0.6, -0.3, 0.]
        self.default_dof_pos = np.array(self.default_dof_pos)
        self.obs_scale={
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "ang_vel": 1.,
            "lin_vel_cmd":2.,
            "ang_vel_cmd":1.,
        }
        self.action_scale = 0.5
        self.clip_observation = 100.
        self.clip_action = 100.

        self.prop_obs_history = np.zeros((self.num_prop_obs_input,self.long_history))

        self.last_actions_buf = np.zeros(self.num_actions)
        self.inference_count = 0
        self.phase_count = 0
        self.dt = 0.01
        self.gait_period = 0.7
        self.exp_filter = expFilter(0.6)
        self.bootstrap()


    def bootstrap(self):
        "预热用"
        input_feed = {
            "input": np.zeros(self.num_prop_obs_input * self.long_history)[None,:].astype(np.float32),
        }
        actions = np.squeeze(self.onnx_session.run(["output"], input_feed))

    def get_phase(self):
        phase = self.inference_count * self.dt / self.gait_period
        obs = np.array([np.sin(2. * np.pi * phase),
                        np.cos(2. * np.pi * phase)])
        return obs

    def build_observations(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_euler_angle = obs_group["euler_angle"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"].copy()
        stand = (np.linalg.norm(obs_commands)<0.05)
        obs_commands[...,:2] *= self.obs_scale["lin_vel_cmd"]
        obs_commands[...,2] *= self.obs_scale["ang_vel_cmd"]
        obs_phase = self.get_phase()
        if stand:
            obs_phase[:] = 0
            self.phase_count = 0

        # 本体感知proprioception 47
        prop_obs = np.concatenate((
            obs_phase, # 2
            obs_commands, # 3
            obs_dof_pos, # 12
            obs_dof_vel, # 12
            obs_last_actions, # 12
            obs_base_ang_vel, # 3
            obs_euler_angle, # 3
        ),axis=-1)

        self.prop_obs_history=np.roll(self.prop_obs_history,shift=-1,axis=1)
        self.prop_obs_history[:,-1] = prop_obs

        return self.prop_obs_history
    
    def build_observations_v2(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_euler_angle = obs_group["euler_angle"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"].copy()
        obs_commands[...,:2] *= self.obs_scale["lin_vel_cmd"]
        obs_commands[...,2] *= self.obs_scale["ang_vel_cmd"]
        obs_phase = self.get_phase()

        # 本体感知proprioception 47
        prop_obs = np.concatenate((
            obs_phase, # 3
            obs_commands, # 2
            obs_dof_pos, # 12
            obs_dof_vel, # 12
            obs_last_actions, # 12
            obs_base_ang_vel, # 3
            obs_euler_angle, # 3
        ),axis=-1)

        return prop_obs

    def inference(self, obs_group):
        prop_obs_history = self.build_observations(obs_group)
        obs_input = prop_obs_history.transpose().flatten() # [num_obs, history] -> [obs1,obs2,...,obsN]

        input_feed = {
            "input": obs_input[None,:].astype(np.float32),
        }
        actions = np.squeeze(self.onnx_session.run(["output"], input_feed))
        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf[:] = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        self.inference_count += 1
        self.phase_count += 1
        return dof_pos_target_urdf
    
    def reset(self):
        self.prop_obs_history = np.zeros((self.num_prop_obs_input,self.long_history))
        self.last_actions_buf = np.zeros(self.num_actions)
        # norminal_obs_group = {
        #     "dof_pos":np.zeros(12),
        #     "dof_vel":np.zeros(12),
        #     "angular_velocity":np.zeros(3),
        #     "projected_gravity":np.array([0.,0.,-1.]),
        #     "commands":np.zeros(3),
        #     "yaw_delta":np.zeros(2,dtype=float),
        # }
        # norminal_obs =self.build_observations_v2(norminal_obs_group)
        # self.prop_obs_history[:,:] = norminal_obs[:,None]
        self.exp_filter.reset()
        self.inference_count = 0
        self.phase_count = 0

if __name__=="__main__":
    a=humanoid_walk_onnx_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py/policy/model.onnx")
    a.reset()
    obs_group={
        "dof_pos":np.zeros(12),
        "dof_vel":np.zeros(12),
        "angular_velocity":np.zeros(3),
        "euler_angle":np.zeros(3),
        "commands":np.zeros(3),
    }
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    for i in range(100):
        print(a.inference(obs_group))
