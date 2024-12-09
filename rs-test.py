from datetime import datetime
import argparse
import os
import time
import shutil
import json
import copy
from arguments import add_args
from tensorboardX import SummaryWriter
from util import *
from config.main_config import rllib_env_config
from envs.roadmap_env.MultiAgentRoadmapEnv import MultiAgentRoadmapEnv
from envs.roadmap_env.DijkstraRoadmapEnv import DijkstraRoadmapEnv
from envs.roadmap_env.DummyEnv import DummyEnv
from Agent.RainbowAgent import DQNAgent

def main(args):
    test, debug = args.test, args.debug

    if args.dataset == 'NY':
        args.roadmap_dir = 'drive_service'  # 暂时没用
        args.config_dir = 'NY'
    elif args.dataset == 'manhattan':
        args.roadmap_dir = 'manhattan'  # 暂时没用
        args.config_dir = 'manhattan'

    # 设置输出路径
    if not test:
        timestr = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        output_dir = args.output_dir + '/' + timestr
        # if args.n_rollout_threads != 32:
        #     output_dir += f'_threads={args.n_rollout_threads}'

        if args.share_parameter:
            output_dir += '_ShareParam'
        if args.type2_act_dim != 10:
            output_dir += f'_Act{args.type2_act_dim}'
        if args.gr != 200:
            output_dir += f'_GR={args.gr}'
        if args.batch_size != 64:
            output_dir += f'_BatchSize={args.batch_size}'
        if args.net_width != 150:
            output_dir += f'_NetWidth={args.net_width}'

        if args.num_car is not None:
            output_dir += f'_NU={args.num_car}'
        if args.num_order is not None:
            output_dir += f'_UH={args.num_order}'

        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        output_dir += f'_embedDim512_decay500000_epsilon_envType{args.env_type}'
        # _embedDim512_decay500000_Dijkstra
        writer = SummaryWriter(log_dir=output_dir)

        args.output_dir = output_dir
    else:
        writer = None

    # 设置配置文件

    my_env_config = args.config_dir
    rllib_env_config['my_env_config'] = my_env_config
    rllib_env_config['args'] = args
    # 将参数配置写入输出路径
    if not test:
        tmp_dict = copy.deepcopy(rllib_env_config)
        tmp_dict['args'] = vars(tmp_dict['args'])  # Namespaceargsdictparams.json
        tmp_dict['setting_dir'] = args.setting_dir
        with open(os.path.join(output_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(tmp_dict))
    # 环境和相关参数
    envs = MultiAgentRoadmapEnv(rllib_env_config)
    Max_train_steps = args.Max_train_steps
    random_seed = args.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # 智能体相关参数
    agent = DQNAgent(envs, memory_size=1000, batch_size=64, target_update=2, seed=1)
    agent.train(Max_train_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print(args)

    if not args.test and args.cuda:
        print(f'choose use gpu {args.gpu_id}...')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print('choose use cpu...')
        device = torch.device("cpu")
    main(args)