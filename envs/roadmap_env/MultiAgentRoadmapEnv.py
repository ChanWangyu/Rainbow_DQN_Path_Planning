# numpy.arraydf
import copy
import json
import numpy as np
import osmnx
import networkx as nx
from envs.roadmap_env.BaseRoadmapEnv import BaseRoadmapEnv
from envs.entities.state import JointState
from util import *
from envs.entities.action_space import car_env_actions


# def print(*args, **kwargs):  #
#     pass

class MultiAgentRoadmapEnv(BaseRoadmapEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        # print("env_config: ", env_config)

    def get_not_visited_adj_nodes(self, car_position, car_id):
        # Get adjacent nodes that have not been visited
        adj_nodes = list(nx.neighbors(self.G, car_position))
        tmp_adj_nodes = copy.deepcopy(adj_nodes)
        # print(adj_nodes, len(adj_nodes))
        if len(adj_nodes) == 0:
            return adj_nodes
        for index in range(len(tmp_adj_nodes)):
            # print(np.isin(tmp_adj_nodes[index], self.saved_car_IdTrajs[car_id]))
            if np.isin(tmp_adj_nodes[index], self.saved_car_IdTrajs[car_id]):
                adj_nodes.remove(tmp_adj_nodes[index])
        # print(self.saved_car_IdTrajs, adj_nodes)
        return adj_nodes

    def get_direction(self, car_position_id, adj_position_id_list):
        adj_direction = []
        car_x, car_y = self.G.nodes[car_position_id]['x'], self.G.nodes[car_position_id]['y']
        for adj in adj_position_id_list:
            tmp_adj_x, tmp_adj_y = self.G.nodes[adj]['x'], self.G.nodes[adj]['y']
            angle = azimuthAngle(car_x, car_y, tmp_adj_x, tmp_adj_y)
            direction = angle2direction(angle)
            adj_direction.append(direction)
        return adj_direction

    def get_action_mask(self):
        # 找到当前节点的相邻节点并计算相关角度并覆盖的方向，所有相邻节点未覆盖的方向将会给予mask或者更多负奖励，
        # 并且返回mask和相邻节点的角度
        num_action = 8
        mask = np.zeros((self.num_car, num_action))

        for i in range(len(self.cars)):
            car_position = self.cars[i].position_id
            adjacent_nodes = self.get_not_visited_adj_nodes(car_position, i)
            # print('adjacent_nodes',adjacent_nodes)
            adjacent_direction = self.get_direction(car_position, adjacent_nodes)
            mask[i][adjacent_direction] = 1
            self.adj_nodes[i] = self.adj_nodes[i] + adjacent_nodes
            self.adj_direction[i] = self.adj_direction[i] + adjacent_direction
            for adj in adjacent_nodes:
                # tmp_adj_x, tmp_adj_y = self.rm.lonlat2pygamexy(self.G.nodes[adj]['x'], self.G.nodes[adj]['y'])
                self.adj_lonlat[i].append((self.G.nodes[adj]['x'], self.G.nodes[adj]['y']))
                tmp_dist = nx.shortest_path_length(self.G, car_position, adj, weight='length')
                self.adj_dist[i].append(tmp_dist)
            # print("src_node: ", src_node, '\n', "pairs: ", pairs, '\n', "near_set: ", near_set)
        # print('adj_dist:', self.adj_dist)
        # print("adj_nodes: ", self.adj_nodes)
        # print("adj_direction: ", self.adj_direction)
        # print("adj_lonlat: ", self.adj_lonlat)
        # print('mask: ', mask)
        return mask

    def reset(self):
        obs = super().reset()
        # print(obs)
        mask = self.get_action_mask()
        # print(obs, '\n', self.get_mask())
        return obs, mask


    def step(self, action_dict):
        # action_dict{'uav1': [0.5, 0.2], 'uav2': [0.1, 0.9], 'car1': 5}
        reward = super().step(action_dict)
        done =False
        next_state = torch.tensor([])
        info = 0
        # print(car.position_id)
        for i in range(len(self.cars)):
            car_position = self.cars[i].position_id
            order_drop_off_position = self.orders[i].drop_off_position_id
            adj_nodes = self.get_not_visited_adj_nodes(car_position, i)
            reward = - np.linalg.norm(np.array(self.cars[i].position) - np.array(self.orders[i].drop_off_position))
            # print(reward)
            if car_position == order_drop_off_position or len(adj_nodes) == 0:
                done = True
            else:
                done= False
            if car_position == order_drop_off_position:
                print('arrived')
                print(f'saved_car_trajs_{len(self.saved_car_IdTrajs[0])}: ', self.saved_car_IdTrajs)
                dijsktra_path = nx.shortest_path(self.G, self.saved_car_IdTrajs[0][0], self.saved_car_IdTrajs[0][-1])
                info = len(self.saved_car_IdTrajs[0]) - len(dijsktra_path)
                print(info)
            # next_state = torch.tensor(np.concatenate(
            #     (self.graph_embed[self.cars[0].position_id], self.graph_embed[self.orders[0].drop_off_position_id])
            # ))
            next_state = torch.tensor(np.concatenate(
            (self.graph_embed[self.nodes.index(self.cars[0].position_id)],
             self.graph_embed[self.nodes.index(self.orders[0].drop_off_position_id)])
        ))
        # if len(self.saved_car_IdTrajs[0]) % 2 == 0:
        #     print(f'saved_car_trajs_{len(self.saved_car_trajs[0])}: ', self.saved_car_trajs)
        return next_state, self.get_action_mask(), reward, done, info





