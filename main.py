from env import GridmapEnv
from Agent.RainbowAgent import DQNAgent
import numpy as np
import torch

def main():
    # 初始化环境与随机种子
    seed = 42
    env = GridmapEnv(seed=seed)  # 确保你导入了正确的Gridmap环境
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 超参数设置
    memory_size = 10000
    batch_size = 32
    target_update = 100
    gamma = 0.99
    alpha = 0.2
    beta = 0.6
    prior_eps = 1e-6
    v_min = 0.0
    v_max = 200.0
    atom_size = 51
    n_step = 3

    # 创建DQN代理
    agent = DQNAgent(
        env=env,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update=target_update,
        seed=seed,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        prior_eps=prior_eps,
        v_min=v_min,
        v_max=v_max,
        atom_size=atom_size,
        n_step=n_step,
    )

    # 开始训练
    num_episodes = 1000  # 设置合适的训练轮数
    plotting_interval = 100  # 每100轮绘制一次结果
    agent.train(num_episode=num_episodes, plotting_interval=plotting_interval)

    # # 测试代理，保存测试视频到文件夹
    # video_folder = "./videos/"
    # agent.test(video_folder=video_folder)


if __name__ == "__main__":
    main()
