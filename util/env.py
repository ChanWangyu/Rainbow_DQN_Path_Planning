import numpy as np
import networkx as nx  # 导入 networkx

# todo:可以添加一个检查网格是否可达的函数

CURRENT_POSITION = -10
END_POSITION = 10
ARRIVE_REWARD = 100
STEP_REWARD = 0
DISTANCE_C = 0 #-0.1

# set random seed
def initialize_seed(seed):
    np.random.seed(seed)

class GridmapEnv():
    """Grid Map Environment for path planning."""

    def __init__(self, grid_size=(10, 10), obstacle_ratio=0.2, seed=None):
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.start = None
        self.end = None
        self.cur = None
        self.done = False

        initialize_seed(seed)

        # 初始化 action 和 observation 空间
        self.action_space = [0,1,2,3]  # 上、下、左、右

        self.grid_map = self._create_grid_map()
        self._reset_start_end()

    def _create_grid_map(self):
        """创建包含障碍物的随机地图。"""
        # np.random.seed(self.seed)
        grid_map = np.zeros(self.grid_size, dtype=int)
        num_obstacles = int(self.grid_size[0] * self.grid_size[1] * self.obstacle_ratio)

        obstacles = np.random.choice(self.grid_size[0] * self.grid_size[1], num_obstacles, replace=False)
        for obstacle in obstacles:
            x = obstacle // self.grid_size[1]
            y = obstacle % self.grid_size[1]
            grid_map[x, y] = 1  # 1 代表障碍物

        return grid_map

    def _reset_start_end(self):
        """确保起点和终点不在障碍物上。"""
        self.start = self._get_random_point(exclude_obstacles=True)
        self.end = self._get_random_point(exclude_obstacles=True)
        self.cur = np.array(self.start)  # 初始化状态

    def _get_random_point(self, exclude_obstacles=False):
        """在地图上选取一个随机点。"""
        while True:
            point = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            if not exclude_obstacles or self.grid_map[point] == 0:
                return point

    def _get_full_state(self):
        """生成包含当前位置和目标的完整状态矩阵。"""
        full_state = self.grid_map.copy()
        full_state[self.cur[0], self.cur[1]] = CURRENT_POSITION  # 标记当前位置
        full_state[self.end[0], self.end[1]] = END_POSITION  # 标记目标位置
        return full_state

    def reset(self):
        """重置环境，返回初始观测值和动作掩码。"""
        # 重新生成地图，并重置起点和终点
        self.grid_map = self._create_grid_map()
        self._reset_start_end()

        self.done = False  # 重置完成标志
        obs = self._get_full_state()  # 获取完整初始状态
        mask = self._get_action_mask()  # 获取动作掩码

        # return obs, mask
        return obs

    def restart(self):
        """重新开始，更换起终点"""
        self.cur = self.start
        self.done = False
        obs = self._get_full_state()
        mask = self._get_action_mask()

        return obs, mask

    def step(self, action):
        """执行一步操作，并返回状态矩阵、动作掩码、奖励、是否完成和信息。"""
        x, y = self.cur

        # 根据动作更新位置
        # if action == 0 and x > 0 and self.grid_map[x - 1, y] == 0:
        #     x -= 1  # 向上移动
        # elif action == 1 and x < self.grid_size[0] - 1 and self.grid_map[x + 1, y] == 0:
        #     x += 1  # 向下移动
        # elif action == 2 and y > 0 and self.grid_map[x, y - 1] == 0:
        #     y -= 1  # 向左移动
        # elif action == 3 and y < self.grid_size[1] - 1 and self.grid_map[x, y + 1] == 0:
        #     y += 1  # 向右移动

        # self.cur = np.array([x, y])
        # done = (x, y) == self.end

        next_x, next_y = self.cur
        if action == 0:
            next_x -= 1
        elif action == 1:
            next_x += 1
        elif action == 2:
            next_y -= 1
        elif action == 3:
            next_y += 1

        if next_x < 0 or next_x >= self.grid_size[0] or next_y < 0 or next_y >= self.grid_size[1]:
            reward = -10
            done = False
        elif self.grid_map[next_x, next_y] != 0:
            reward = -10
            done = False
        else:
            self.cur = np.array([next_x, next_y])
            done = (next_x, next_y) == self.end

            # 奖励：到达终点为1，否则根据距离给负奖励
            distance_to_goal = np.linalg.norm(np.array([x, y]) - np.array(self.end))
            reward = ARRIVE_REWARD if done else DISTANCE_C * distance_to_goal+STEP_REWARD

        actual_steps = np.linalg.norm(self.cur - np.array(self.start))
        info = actual_steps
        # if done:
        #     print("Arrived at the goal!")

        next_state = self._get_full_state()
        # return next_state, self._get_action_mask(), reward, done, info
        return next_state, reward, done, info

    def _get_action_mask(self):
        """生成动作掩码，避免选择无效动作。"""
        x, y = self.cur
        mask = [1, 1, 1, 1]  # 上、下、左、右
        if x == 0 or self.grid_map[x - 1, y] == 1:
            mask[0] = 0  # 无法向上移动
        if x == self.grid_size[0] - 1 or self.grid_map[x + 1, y] == 1:
            mask[1] = 0  # 无法向下移动
        if y == 0 or self.grid_map[x, y - 1] == 1:
            mask[2] = 0  # 无法向左移动
        if y == self.grid_size[1] - 1 or self.grid_map[x, y + 1] == 1:
            mask[3] = 0  # 无法向右移动
        # return mask
        return np.array(mask, dtype=np.int32)


def main():
    env = GridmapEnv(seed=10)
    obs, mask = env.reset()
    print("Initial Observation:")
    print(obs)
    print("Action Mask:", mask)

    obs, mask = env.reset()
    print("Initial Observation:")
    print(obs)
    print("Action Mask:", mask)

    print(env.action_space)
    print(env.grid_size)

    next_state, mask, reward, terminated, info = env.step(1)
    done = terminated
    print(type(next_state), type(mask), type(reward), type(done), type(info))

if __name__ == '__main__':
    main()