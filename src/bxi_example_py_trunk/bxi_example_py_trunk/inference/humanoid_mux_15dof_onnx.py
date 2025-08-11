# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
from bxi_example_py_trunk.inference.base_agent import baseAgent
from bxi_example_py_trunk.inference.humanoid_dh_long_onnx import humanoid_dh_long_onnx_Agent
from bxi_example_py_trunk.inference.humanoid_comp_15dof_onnx import humanoid_comp_15dof_onnx_Agent
import numpy as np
from copy import deepcopy

class humanoid_mux_15dof_onnx_Agent(baseAgent):
    def __init__(self, policy_path_main, policy_path_comp):
        self.agent1 = humanoid_dh_long_onnx_Agent(policy_path_main)
        self.agent2 = humanoid_comp_15dof_onnx_Agent(policy_path_comp)

    def bootstrap(self):
        self.agent1.bootstrap()
        self.agent2.bootstrap()

    def inference(self, obs_group):
        main_agent_obs_group = deepcopy(obs_group)
        main_agent_obs_group["dof_pos"] = obs_group["dof_pos"][3:15]
        main_agent_obs_group["dof_vel"] = obs_group["dof_vel"][3:15]
        out1 = self.agent1.inference(main_agent_obs_group)

        out2 = self.agent2.inference(obs_group)
        out2[3:15] = 0

        added_out = out2
        added_out[3:15] += out1
        
        return added_out
    
    def reset(self):
        self.agent1.reset()
        self.agent2.reset()

if __name__=="__main__":
    a=humanoid_mux_15dof_onnx_Agent("/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/Aug07_12-29-51_model_522500.onnx",
                                    "/home/xuxin/allCode/bxi_ros2_example/src/bxi_example_py_trunk/policy/Aug07_12-29-51_model_522500_compensate.onnx")
    obs_group={
        "dof_pos":np.zeros(15),
        "dof_vel":np.zeros(15),
        "angular_velocity":np.zeros(3),
        "projected_gravity":np.array([0.,0.,-1.]),
        "commands":np.zeros(3),
    }
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    for i in range(100):
        print(a.inference(obs_group))
