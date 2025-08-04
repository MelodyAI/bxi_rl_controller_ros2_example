import numpy as np
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.inference.exp_filter import expFilter
import time
import onnxruntime as ort

class humanoid_dh_long_comp_onnx_Agent(baseAgent):
    def __init__(self, policy_path):
        self.num_prop_obs_input = 47
        self.num_estimator_input = 42
        self.include_history_steps = 5
        self.long_history = 64
        self.num_actions = 12
        # policy_path = "/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/Aug03_17-25-15_model_524000.onnx"
        policy_path = "/home/bxi/bxi_ws_xx/src/bxi_example_py_trunk/policy/Aug03_17-25-15_model_524000.onnx"

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
        
        # dof_pos归一化范围
        dof_pos_limits=[[-0.7850,  0.7850],
                        [-0.5230,  0.7850],
                        [-2.0000,  1.0000],
                        [ 0.0000,  2.3550],
                        [-2.0000,  2.0000],
                        [-0.3490,  0.3490],
                        [-0.7850,  0.7850],
                        [-0.7850,  0.5230],
                        [-2.0000,  1.0000],
                        [ 0.0000,  2.3550],
                        [-2.0000,  2.0000],
                        [-0.3490,  0.3490]]
        self.dof_pos_limits=np.array(dof_pos_limits)

        self.default_dof_pos = [0., 0., -0.3, 0.6, -0.3, 0.,
                                0., 0., -0.3, 0.6, -0.3, 0.]
        self.default_dof_pos = np.array(self.default_dof_pos)

        l_low=self.default_dof_pos-self.dof_pos_limits[:,0]
        l_up=self.dof_pos_limits[:,1]-self.default_dof_pos
        dof_scale=1.0/np.max(np.array([l_low,l_up]),axis=0)

        self.obs_scale={
            "dof_pos": dof_scale,
            "dof_vel": 0.1,
            "ang_vel": 0.1,
        }
        self.action_scale = 1.0
        self.clip_observation = 100.
        self.clip_action = 100.

        self.prop_obs_history = np.zeros((self.num_prop_obs_input,self.long_history))
        self.estimator_obs_history = np.zeros((self.num_estimator_input,self.include_history_steps))   

        self.last_actions_buf = np.zeros(self.num_actions)
        self.inference_count = 0
        self.dt = 0.01
        self.gait_period = 0.6
        self.exp_filter = expFilter(0.6)
        
        # comp_policy_path = "/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/Aug03_17-25-15_model_524000_compensate.onnx"
        comp_policy_path = "/home/bxi/bxi_ws_xx/src/bxi_example_py_trunk/policy/Aug03_17-25-15_model_524000_compensate.onnx"
        self.onnx_session_comp = ort.InferenceSession(
            comp_policy_path,
            providers=providers,
            sess_options=options
        )
        print([ipt.name for ipt in self.onnx_session_comp.get_inputs()])

        self.bootstrap()


    def bootstrap(self):
        "预热用"
        obs_group={
            "dof_pos":np.zeros(12,dtype=float),
            "dof_vel":np.zeros(12,dtype=float),
            "angular_velocity":np.zeros(3,dtype=float),
            "commands":np.zeros(3,dtype=float),
            "projected_gravity":np.array([0.,0.,-1.],dtype=float),
            "yaw_delta":np.zeros(2,dtype=float),
        }
        self.inference(obs_group)

    def get_phase(self):
        phase = self.inference_count * self.dt / self.gait_period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase= np.cos(2 * np.pi * phase)
        obs = np.array([sin_phase,cos_phase])
        return obs

    def build_observations(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"]
        # obs_commands[...,2] *= self.obs_scale["ang_vel"]
        obs_commands[...,2] = obs_group["yaw_delta"][0] * self.obs_scale["ang_vel"] # 使用导航角计算转向

        x_vel = obs_commands[0]
        if x_vel < 0.4: # 特别小的指令改成站立
            x_vel = 0.
        if x_vel > 0.8: # clip
            x_vel = 0.8
        obs_commands[0] = x_vel
        print(obs_commands)
        
        obs_phase = self.get_phase()

        # 本体感知proprioception 47
        prop_obs = np.concatenate((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
            obs_commands,
            obs_phase
        ),axis=-1)

        # estimator 42
        estimator_obs = np.concatenate((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
        ),axis=-1)

        self.prop_obs_history=np.roll(self.prop_obs_history,shift=-1,axis=1)
        self.prop_obs_history[:,-1] = prop_obs

        self.estimator_obs_history=np.roll(self.estimator_obs_history,shift=-1,axis=1)
        self.estimator_obs_history[:,-1] = estimator_obs

        return self.prop_obs_history, self.estimator_obs_history, prop_obs

    def build_observations_v2(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"]
        obs_commands[...,2] *= self.obs_scale["ang_vel"]
        obs_phase = self.get_phase()

        # 本体感知proprioception 47
        prop_obs = np.concatenate((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
            obs_commands,
            obs_phase
        ),axis=-1)

        # estimator 42
        estimator_obs = np.concatenate((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
        ),axis=-1)

        return prop_obs, estimator_obs

    def inference(self, obs_group):
        prop_obs_history, estimator_obs_history, prop_obs_comp = self.build_observations(obs_group)

        input_feed = {
            "prop_obs": prop_obs_history.flatten()[None,:].astype(np.float32),
            "estimator_obs": estimator_obs_history.flatten()[None,:].astype(np.float32),
        }
        actions = np.squeeze(self.onnx_session.run(["output"], input_feed))
        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        main_agent_out = actions * self.action_scale + self.default_dof_pos

        input_feed_comp = {"prop_obs": prop_obs_comp[None,:].astype(np.float32)}
        comp_out = np.squeeze(self.onnx_session_comp.run(["output"], input_feed_comp))

        dof_pos_target_urdf = main_agent_out + comp_out

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        self.inference_count += 1
        return dof_pos_target_urdf
    
    def reset(self):
        self.last_actions_buf = np.zeros(self.num_actions)
        norminal_obs_group = {
            "dof_pos":np.zeros(12),
            "dof_vel":np.zeros(12),
            "angular_velocity":np.zeros(3),
            "projected_gravity":np.array([0.,0.,-1.]),
            "commands":np.zeros(3),
            "yaw_delta":np.zeros(2,dtype=float),
        }
        norminal_obs, estimator_obs =self.build_observations_v2(norminal_obs_group)
        self.prop_obs_history[:,:] = norminal_obs[:,None]
        self.estimator_obs_history[:,:] = estimator_obs[:,None] 
        self.exp_filter.reset()
        self.inference_count = 0

if __name__=="__main__":
    a=humanoid_dh_long_comp_onnx_Agent("")
    obs_group={
        "dof_pos":np.zeros(12),
        "dof_vel":np.zeros(12),
        "angular_velocity":np.zeros(3),
        "projected_gravity":np.array([0.,0.,-1.]),
        "commands":np.zeros(3),
        "yaw_delta":np.zeros(2,dtype=float),
    }
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    for i in range(100):
        print(a.inference(obs_group))
