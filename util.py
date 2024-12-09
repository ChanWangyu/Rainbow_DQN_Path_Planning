import math
import pandas as pd
import torch
import numpy as np
import importlib

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def import_env_config(config_dir):
    # 创建一个模块规范对象
    spec = importlib.util.spec_from_file_location('config', config_dir)
    if spec is None:
        print('Config file not found.')
        exit(0)
    # 创建模块对象
    config = importlib.util.module_from_spec(spec)
    # 执行模块代码
    spec.loader.exec_module(config)
    # print("config: ", config, "type of config: ", type(config))
    env_config = config.env_config
    return env_config


def fillin_lazy_args(args, dataset_str):
    # print(args.config_dir, args.setting_dir, args.roadmap_dir)
    if hasattr(args, 'config_dir'):
        args.config_dir = f'config/roadmap_config/{dataset_str}/env_config_' + args.config_dir + '.py'
    if hasattr(args, 'setting_dir'):
        args.setting_dir = f'src/envs/roadmap_env/setting/{dataset_str}/' + args.setting_dir
    if hasattr(args, 'roadmap_dir'):
        args.roadmap_dir = f'data/{dataset_str}'
    # print(args.config_dir, args.setting_dir, args.roadmap_dir)
    return args


def li2di(a, agent_name_list):
    action_dict = {}
    for i, name in enumerate(agent_name_list):
        action_dict[name] = a[i]
    return action_dict


def li2di_vec(a, agent_name_list):
    n_rollout_threads = len(a)
    action_dict = []
    for e in range(n_rollout_threads):
        action_d = {}
        for i, name in enumerate(agent_name_list):
            action_d[name] = a[e][i]
        action_dict.append(action_d)
    return action_dict


def di2li(*args):
    out = [list(arg.values()) for arg in args]
    return out if len(out) >= 2 else out[0]


def di2li_vec(*args):
    # for arg in args:
    #     print(type(arg))  # np.array
    # print(args)
    # print("arg: " + arg for arg in args)
    out = [[list(arg[i].values()) for i in range(len(arg))] for arg in args]
    # print(len(out), len(out[0]), len(out[0][0]))
    return out if len(out) >= 2 else out[0]


def tensor_element_unique(tensor, q_tensor):
    # method 1
    # unique, inverse_indices = torch.unique(tensor, return_inverse=True)
    # repeated_indices = torch.nonzero(inverse_indices == torch.arange(len(tensor)).unsqueeze(1)).squeeze()
    unique, counts = torch.unique(tensor, return_counts=True)
    repeated_indices = torch.nonzero(counts > 1).squeeze()
    for index in repeated_indices:
        q_index = torch.where(tensor == unique[index])


def max_greedy_action(q_net, length, max_index):
    action = torch.full((1, length), max_index, device=device).squeeze()
    q_net = torch.where(torch.isinf(q_net), torch.tensor(float('-inf'), device=device), q_net)
    while True:
        max_val = torch.max(q_net)
        indices = torch.where(q_net == max_val)
        indices_col = np.random.randint(indices[0].shape[0])
        row = indices[0][indices_col]
        col = indices[1][indices_col]
        if max_val != float('-inf'):
            action[row] = col
            q_net[row] = float('-inf')
            q_net[:, col] = float('-inf')
            # q_net = torch.cat([q_net[:row, :], q_net[row+1:, :]])
            # q_net = torch.cat([q_net[:, :col], q_net[:, col+1:]], dim=1)
        else:
            break
    # print(action)
    return action.unsqueeze(0)


def random_action(q_net, length, max_index):
    action = torch.full((1, length), max_index).squeeze()
    while True:
        indices = torch.where((q_net != float('-inf')) & (q_net != float('inf')))
        # print(indices)
        if min(indices[0].shape) == 0:
            break
        indices_col = np.random.randint(indices[0].shape[0])
        # print(indices_col)
        row = indices[0][indices_col]
        col = indices[1][indices_col]
        action[row] = col
        q_net[row] = float('-inf')
        q_net[:, col] = float('-inf')
        # q_net = torch.cat([q_net[:row, :], q_net[row+1:, :]])
        # q_net = torch.cat([q_net[:, :col], q_net[:, col+1:]], dim=1)
    # print(action)
    return action.unsqueeze(0)


def get_direction(car_gird_position, order_gird_position):
    if car_gird_position[0] < order_gird_position[0]:
        direction_long = np.array([0, 1])
    elif car_gird_position[0] > order_gird_position[0]:
        direction_long = np.array([1, 0])
    else:
        direction_long = np.array([0, 0])
    if car_gird_position[1] < order_gird_position[1]:
        direction_lat = np.array([0, 1])
    elif car_gird_position[1] > order_gird_position[1]:
        direction_lat = np.array([1, 0])
    else:
        direction_lat = np.array([0, 0])
    # print('direction_long: ', direction_long, 'direction_lat: ', direction_lat)
    direction = np.concatenate((direction_long, direction_lat))
    # print(direction, ',', 'shape of direction:', direction.shape)
    return direction


def azimuthAngle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


def angle2direction(angle):
    direction = 0
    if 0.0 <= angle < 22.5 or 337.5 <= angle < 360.0 :
        direction = 0
    elif 22.5 <= angle < 67.5:
        direction = 1
    elif 67.5 <= angle < 112.5:
        direction = 2
    elif 112.5 <= angle < 157.5:
        direction = 3
    elif 157.5 <= angle < 202.5:
        direction = 4
    elif 202.5 <= angle < 247.5:
        direction = 5
    elif 247.5 <= angle < 292.5:
        direction = 6
    else:
        direction = 7
    return direction


if __name__ == '__main__':
    pass

