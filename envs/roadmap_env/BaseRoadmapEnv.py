import copy
import random

import osmnx.distance
import torch
import numpy as np
import osmnx
import networkx as nx
import math
import os
import os.path as osp
import pandas as pd
import gym
from gym import spaces
from datetime import datetime
from config.model_config import model_config
from envs.entities.state import OrderState, CarState, JointState
from envs.roadmap_env.roadmap_utils import Roadmap
from envs.entities.utils import *

project_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__))))  # base.py
# print(project_dir)
# def print(*args, **kwargs):  #
#     pass

count = 0

class BaseRoadmapEnv:
    def __init__(self, rllib_env_config):
        # print("base_rl_env: ", rllib_env_config)
        global count
        print(f'{count}env...')
        count += 1

        self.args = rllib_env_config['args']
        env_config = rllib_env_config['my_env_config']
        self.env_config = env_config
        self.output_dir = self.args.output_dir
        self.order_data = pd.read_csv(f'data/order/2015-01-01-18-19_mean_order.csv')
        self.car_data = pd.read_csv(f'data/car/car_data.csv')
        # debug
        self.reward_scale = self.args.reward_scale

        # == roadmap
        self.roadmap_dir = self.args.roadmap_dir

        # self.G = osmnx.load_graphml(f'data/random_graph.graphml')
        # self.graph_embed = np.load(f'envs/graph_embed/embed_data/random_node_embed.npy')
        self.G = osmnx.load_graphml(f'data/manhattan/undirected_partial_road_network.graphml')
        self.graph_embed = np.load(f'envs/graph_embed/embed_data/partial_node_embed.npy')
        self.nodes = list(self.G.nodes)
        # # validate
        self.num_order = 1
        self.num_car = 1
        self.num_agent = self.num_car
        order_state_dim = 128
        car_state_dim = 128
        self.obs_dim = order_state_dim * self.num_order + car_state_dim * self.num_car

        # obs and act spaces
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,))
        self.share_observation_space = self.observation_space
        # self.mappo_share_observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim * self.num_agent,))  # mappoconcatccobs
        self.car_action_space = spaces.Discrete(8)

        self.device = torch.device('cpu')

        # --- for noadmap ---
        # timestepactiondst(lon,lat)
        self.wipe_adj_things()
        self.car_cur_node = [None for _ in range(self.num_car)]  #

        # --- some variable(including entities) which change during explore, so they **must** be reset in self.reset
        self.cars = []
        self.orders = []
        self.saved_car_trajs = None
        self.saved_car_IdTrajs = None
        self.saved_car_trajs_len = None
        # roadmap

        # debug
        '''variables that used in self.callback'''
        self.best_train_reward = -float('inf')
        self.best_eval_reward = -float('inf')
        # print(self.num_car, self.num_order)

    def set_device(self, device):
        self.device = device

    def wipe_adj_things(self):
        self.adj_nodes = [[] for _ in range(self.num_car)]
        self.adj_direction = [[] for _ in range(self.num_car)]
        self.adj_dist = [[] for _ in range(self.num_car)]
        self.adj_lonlat = [[] for _ in range(self.num_car)]

    def reset_orders(self):
        self.orders = []
        for idx in range(self.num_order):
            # num = random.randint(0, 99)
            num = 42423514
            init_start_node = num
            init_end_node = num
            start_x, start_y = self.G.nodes[init_start_node]['x'], self.G.nodes[init_start_node]['y']
            end_x, end_y = self.G.nodes[init_end_node]['x'], self.G.nodes[init_end_node]['y']
            start = [start_x, start_y]
            end = [end_x, end_y]
            self.orders.append(OrderState(start, end, init_start_node, init_end_node, 0, self.env_config))

    def reset_cars(self):
        self.cars = []
        for idx in range(self.num_car):
            # init_postion_node = random.randint(0, 99)
            init_postion_node = 42421828
            px, py = self.G.nodes[init_postion_node]['x'], self.G.nodes[init_postion_node]['y']
            position = [px, py]
            self.cars.append(CarState(position, init_postion_node, self.env_config))


    def flatten_state(self, state):
        batch_dim = 1  # rllibbatch
        state_list = []
        for s in state:
            state_list.append(s.view(batch_dim, -1))

        flatted_state = torch.cat(state_list, dim=1).to(self.device)
        flatted_state = flatted_state.squeeze(0)  # (x, )squeeze
        return flatted_state.numpy()

    def reset(self):
        # ===step1.reset the **must** entities
        self.wipe_adj_things()
        self.global_timestep = 0
        self.reset_cars()
        self.reset_orders()
        self.saved_car_trajs = [np.empty(shape=(0, 2)) for _ in range(self.num_car)]
        self.saved_car_IdTrajs = [np.array([]) for _ in range(self.num_car)]
        for id in range(self.num_car):
            self.saved_car_trajs[id] = np.concatenate((self.saved_car_trajs[id],
                                                       np.array([[self.cars[id].position[0],
                                                               self.cars[id].position[1]]])))
            self.saved_car_IdTrajs[id] = np.append(self.saved_car_IdTrajs[id], self.cars[id].position_id)
        self.saved_car_trajs_len = []  # poisinr

        # roadmap
        state = torch.tensor(np.concatenate(
            (self.graph_embed[self.nodes.index(self.cars[0].position_id)],
             self.graph_embed[self.nodes.index(self.orders[0].drop_off_position_id)])
        ))
        # print("state: ", state)
        return state

    def _movement(self, car_dst_dict, rewards):
        for j, (agent_name, (px, py)) in enumerate(car_dst_dict.items()):
            obj = self.cars[j]
            obj.position[0] = px
            obj.position[1] = py
            # velocity = car_length_dict[agent_name] / self.env_config['tm']
            rewards[agent_name] = 0
            # print('obj:', obj.position, px, py)
        for id in range(self.num_car):
            self.saved_car_trajs[id] = np.concatenate((self.saved_car_trajs[id],
                                                       np.array([[self.cars[id].position[0],
                                                                  self.cars[id].position[1]]])))
            self.saved_car_IdTrajs[id] = np.append(self.saved_car_IdTrajs[id], self.cars[id].position_id)
        return rewards

    def step(self, action):
        # action_dict{'uav1': 12, 'uav2': 4}
        # action_dict{'uav1': [0.5, 0.2], 'uav2': [0.1, 0.9]}
        self.global_timestep += 1
        reward = 0
        self.cars[0].position_id = self.adj_nodes[0][self.adj_direction[0].index(action)]
        self.cars[0].position[0], self.cars[0].position[1] = self.adj_lonlat[0][self.adj_direction[0].index(action)]
        self.wipe_adj_things()
        for id in range(self.num_car):
            self.saved_car_trajs[id] = np.concatenate((self.saved_car_trajs[id],
                                                       np.array([[self.cars[id].position[0],
                                                                  self.cars[id].position[1]]])))
            self.saved_car_IdTrajs[id] = np.append(self.saved_car_IdTrajs[id], self.cars[id].position_id)
        # print('car_direction_dict: ', car_direction_dict)
        # print('car_dst_dict: ', car_dst_dict)
        # == step2. collect data ==
        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    rllib_env_config = {
    'algo_name': None, 'partial_obs': True, 'use_reward_shaping': False,
    'consider_return_to_zero_list': True, 'energy_factor': 0.5,
    'my_env_config': {
        'name': 'NCSU100_2u2c_QOS1', 'num_timestep': 100,
        'theta_range': (0, 6.283185307179586),
        'max_x': 2718, 'max_y': 3255, 'max_z': 100,
        'uav_init_height': 50, 'max_r': 40, 'space_between_building': 100,
        'max_human_movelength': 0, 'max_uav_energy': 1500000, 'max_car_energy': 2000000,
        'num_uav': 2, 'num_car': 2, 'num_human': 100,
        'num_building': 0, 'num_special_human': 10, 'num_special_building': 0,
        'max_data_amount': 2000000000.0,
        'tc': 10, 'tm': 10, 'v_uav': 20, 'v_car': 15,
        'power_tx': 20, 'noise0': -70, 'rho0': -50, 'bandwidth_subchannel': 20000000.0,
        'num_subchannel': 3, 'noise0_density': 5e-20,
        'aA': 2, 'aG': 4, 'nLoS': 0, 'nNLoS': -20, 'psi': 9.6, 'beta': 0.16,
        'p_uav': 3, 'p_poi': 0.1, 'sinr_demand': 1.0,
        'is_uav_competitive': False, 'obs_range': 679.5
    },
    'args': {
        'HID_phi': [0, 0], 'HID_theta': [45, 45], 'K_epochs': 10,
        'Max_train_steps': 1000000.0, 'ModelIdex': 400, 'T_horizon': 2048,
        'W_epochs': 5, 'a_lr': 0.0002, 'batch_size': 64, 'c_lr': 0.0002,
        'clip_rate': 0.2, 'config_dir': 'src/config/roadmap_config/NCSU/env_config_NCSU100_2u2c_QoS1.py',
        'copo_kind': 1, 'cuda': True, 'dataset': 'NCSU', 'debug': False, 'dist': 'Beta', 'entropy_coef': 0.001,
        'entropy_coef_decay': 0.99, 'eoi3_coef': 0.003, 'eoi_coef_decay': 1.0, 'eoi_kind': 3,
        'eval_interval': 100000000000.0, 'gamma': 0.99, 'gpu_id': '0', 'gr': 200, 'hcopo_shift': False,
        'hcopo_shift_513': False, 'hcopo_sqrt2_scale': True, 'l2_reg': 0.001, 'lambd': 0.95,
        'n_rollout_threads': 2, 'nei_dis_scale': 0.25, 'net_width': 150, 'num_subchannel': None,
        'num_test_episode': 1, 'num_uv': None, 'output_dir': '../runs/debug/2023-09-05_08-20-50',
        'reward_scale': 1, 'roadmap_dir': 'data/NCSU/drive', 'save_interval': 200000.0, 'seed': 0,
        'setting_dir': 'src/envs/roadmap_env/setting/NCSU/NCSU100cluster', 'share_layer': False,
        'share_parameter': False, 'sinr_demand': None, 'svo_frozen': False, 'svo_lr': 0.0001, 'test': False,
        'type2_act_dim': 10, 'uav_height': None, 'use_ccobs': False, 'use_copo': False, 'use_eoi': False,
        'use_hcopo': False, 'vf_coef': 1.0, 'write': True
    }
}

    map = BaseRoadmapEnv(rllib_env_config)
    print(map.G)





