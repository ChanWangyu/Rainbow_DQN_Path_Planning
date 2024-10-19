import gym
import numpy as np
import networkx as nx  # 导入 networkx

CURRENT_POSITION = 2
END_POSITION = 3

class GridmapEnv(gym.Env):
    """Grid Map Environment for path planning."""

    def __init__(self, grid_size=(10, 10), obstacle_ratio=0.2, seed=None):
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.start = None
        self.goal = None
        self.cur = None
        self.seed = seed
        self.done = False

        # 初始化 action 和 observation 空间
        self.action_space = gym.spaces.Discrete(4)  # 上、下、左、右
        self.obs_dim = gym.spaces.Box(low=0, high=3, shape=self.grid_size, dtype=np.float32)

        # 初始化图结构
        self.G = nx.Graph()  # 用于最短路径计算
        self.grid_map = self._create_grid_map()
        self._build_graph()
        self._reset_start_end()

    def _create_grid_map(self):
        """创建包含障碍物的随机地图。"""
        np.random.seed(self.seed)
        grid_map = np.zeros(self.grid_size, dtype=int)
        num_obstacles = int(self.grid_size[0] * self.grid_size[1] * self.obstacle_ratio)

        obstacles = np.random.choice(self.grid_size[0] * self.grid_size[1], num_obstacles, replace=False)
        for obstacle in obstacles:
            x = obstacle // self.grid_size[1]
            y = obstacle % self.grid_size[1]
            grid_map[x, y] = 1  # 1 代表障碍物

        return grid_map

    def _build_graph(self):
        """根据地图构建图结构，其中每个非障碍物格子是一个节点。"""
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.grid_map[x, y] == 0:  # 不是障碍物的格子
                    self.G.add_node((x, y))
                    # 添加相邻格子的边（上下左右）
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                            if self.grid_map[nx, ny] == 0:
                                self.G.add_edge((x, y), (nx, ny))

    def _reset_start_end(self):
        """确保起点和终点不在障碍物上。"""
        self.start = self._get_random_point(exclude_obstacles=True)
        self.end = self._get_random_point(exclude_obstacles=True)
        self.state = np.array(self.start)  # 初始化状态

    def _get_random_point(self, exclude_obstacles=False):
        """在地图上选取一个随机点。"""
        while True:
            point = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            if not exclude_obstacles or self.grid_map[point] == 0:
                return point

    def _get_full_state(self):
        """生成包含当前位置和目标的完整状态矩阵。"""
        full_state = self.grid_map.copy()
        full_state[self.state[0], self.state[1]] = CURRENT_POSITION  # 标记当前位置
        full_state[self.end[0], self.end[1]] = END_POSITION  # 标记目标位置
        return full_state

    def reset(self):
        """重置环境，返回初始观测值和动作掩码。"""
        # 重新生成地图，并重置起点和终点
        self.grid_map = self._create_grid_map()
        self._build_graph()
        self._reset_start_end()

        self.done = False  # 重置完成标志
        obs = self._get_full_state()  # 获取完整的初始状态

        mask = self._get_action_mask()  # 获取动作掩码

        return obs, mask  # 返回观测值和动作掩码

    def step(self, action):
        """执行一步操作，并返回状态矩阵、动作掩码、奖励、是否完成和信息。"""
        x, y = self.state

        # 根据动作更新位置
        if action == 0 and x > 0 and self.grid_map[x - 1, y] == 0:
            x -= 1  # 向上移动
        elif action == 1 and x < self.grid_size[0] - 1 and self.grid_map[x + 1, y] == 0:
            x += 1  # 向下移动
        elif action == 2 and y > 0 and self.grid_map[x, y - 1] == 0:
            y -= 1  # 向左移动
        elif action == 3 and y < self.grid_size[1] - 1 and self.grid_map[x, y + 1] == 0:
            y += 1  # 向右移动

        self.state = np.array([x, y])
        done = (x, y) == self.end

        # 奖励：到达终点为1，否则根据距离给负奖励
        distance_to_goal = np.linalg.norm(np.array([x, y]) - np.array(self.end))
        reward = 1 if done else -0.1 * distance_to_goal

        # 计算路径偏差信息
        try:
            shortest_path = nx.shortest_path_length(self.G, source=tuple(self.start), target=tuple(self.end))
            print(shortest_path)
        except nx.NetworkXNoPath:
            shortest_path = float('inf')  # 没有路径时设为无穷大

        actual_steps = np.linalg.norm(self.state - np.array(self.start))
        info = {'path_deviation': actual_steps - shortest_path}

        if done:
            print("Arrived at the goal!")
            print(f"Info: {info}")

        next_state = self._get_full_state()
        return next_state, self._get_action_mask(), reward, done, info

    def _get_action_mask(self):
        """生成动作掩码，避免选择无效动作。"""
        x, y = self.state
        mask = [1, 1, 1, 1]  # 上、下、左、右
        if x == 0 or self.grid_map[x - 1, y] == 1:
            mask[0] = 0  # 无法向上移动
        if x == self.grid_size[0] - 1 or self.grid_map[x + 1, y] == 1:
            mask[1] = 0  # 无法向下移动
        if y == 0 or self.grid_map[x, y - 1] == 1:
            mask[2] = 0  # 无法向左移动
        if y == self.grid_size[1] - 1 or self.grid_map[x, y + 1] == 1:
            mask[3] = 0  # 无法向右移动
        return mask

def main():
    env = GridmapEnv(seed=34)
    obs, mask = env.reset()
    print("Initial Observation:")
    print(obs)
    print("Action Mask:", mask)

    obs, mask, reward, done, info = env.step(1)  # 执行一次动作
    print("Next Observation:")
    print(obs)
    print("Action Mask:", mask)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)


if __name__ == '__main__':
    main()