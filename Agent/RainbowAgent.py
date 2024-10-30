from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
# from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from typer.cli import state
from util.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from util.RainbowNet import CNNNetwork
# from envs.roadmap_env.MultiAgentRoadmapEnv import MultiAgentRoadmapEnv
from util.env import GridmapEnv

class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (util.env): env of grid path planning
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            env: GridmapEnv,
            memory_size: int,
            batch_size: int,
            target_update: int,
            seed: int,
            gamma: float = 0.99,
            # PER parameters
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,
    ):
        obs_dim = env.grid_size
        action_dim = len(env.action_space)

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        # 使用Noisy_net意味着不需要使用Epsilon动作选择

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, action_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, action_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        # v_min和v_max的意义：概率密度函数的横坐标取值
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = CNNNetwork(
            obs_dim[0], action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = CNNNetwork(
            obs_dim[0], action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        self.bad_actions = set()
        self.visited_states = set()

    def select_action(self, state: np.ndarray, mask = None) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        # todo:如果仅仅通过mask来避免动作的话，缺少对网络的惩罚
        q_value = self.dqn(torch.FloatTensor(state).to(self.device).unsqueeze(0).unsqueeze(0))
        # q_value = q_value * torch.tensor(mask).to(self.device)

        for bad_action in self.bad_actions:
            q_value[0, bad_action] = float('-inf')

        selected_action = q_value.argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            # self.transition = [state, mask, selected_action]
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> tuple[Any, Any, Any, Any]:
        """Take an action and return the response of the env."""
        # next_state, mask, reward, done, info = self.env.step(action)
        next_state, reward, done, info = self.env.step(action)

        state_tuple = tuple(next_state.flatten())

        if state_tuple in self.visited_states:
            reward -= 10
            self.bad_actions.add(action.item())
        else:
            self.bad_actions.clear()
            self.visited_states.add(state_tuple)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        # return mask, next_state, reward, done, info
        return next_state, reward, done, info

    def update_model(self, beta: float) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch(beta)
        print(samples["rews"])
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # We are going to combine 1-step loss and n-step loss so as to prevent high-variance.
        # The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(list(indices), new_priorities)

        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss

    def train(self,
              num_map: int,
              num_episode: int,
              showing_interval: int = 2000):
        """Train the agent."""
        self.is_test = False

        update_cnt = 0
        losses = []
        scores = []
        # info_list = []
        score = 0
        # info_total = 0

        for map_index in range(num_map):
            beta = self.beta
            arrive_time = 0
            # state, mask = self.env.reset()
            state = self.env.reset()
            print(state)
            loss = 0
            path = [self.env.cur]
            self.visited_states.clear()

            for episode_idx in range(1, num_episode + 1):
                # action = self.select_action(state, mask)
                action = self.select_action(state)
                # next_mask, next_state, reward, done, info = self.step(action)
                next_state, reward, done, info = self.step(action)

                state = next_state
                # mask = next_mask
                score += reward
                # info_total += info
                path.append(self.env.cur)

                fraction = min(episode_idx / num_episode, 1.0)
                beta = beta + fraction * (1.0 - beta)

                if done:
                    # 重新开始
                    print(f"Map {map_index} - Episode {len(path)} Path: {path}")
                    # state, mask = self.env.restart()
                    state = self.env.restart()
                    self.visited_states.clear()
                    path = [self.env.cur]
                    scores.append(score)
                    score = 0
                    arrive_time += 1
                    # info_list.append(info_total)
                    # info_total = 0

                if self.memory.size >= self.batch_size:
                    loss = self.update_model(beta).item()
                    losses.append(loss)
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # 训练效果的展示
                if episode_idx % showing_interval == 0:
                    # self._plot(episode_idx, scores, losses, info_list)
                    self._print_and_show(map_index, episode_idx, arrive_time, score, loss)

            if arrive_time == 0:
                truncated_path = path[:200]
                print(f"Map {map_index} - Final Path after {num_episode} episodes (first 200 steps): {truncated_path}")

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        # self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            # next_state, mask, reward, done, info = self.step(action)
            next_state, reward, done, info = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)

        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state.unsqueeze(1)).argmax(1)
            next_dist = self.dqn_target.dist(next_state.unsqueeze(1))
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state.unsqueeze(1))
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self) -> None:
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            info_list: List[float],
    ) -> None:
        """Plot the training progresses."""
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('info_list')
        plt.plot(info_list)
        plt.show()

    def _print_and_show(self,
                        map_index: int,
                        episode_idx: int,
                        arrive_times: int,
                        score: int,
                        loss: int) -> None:
        """Print and show the training progresses."""
        print('----------------------------------------------')
        print(f'Map: {map_index}')
        print(f'Episode: {episode_idx}')
        print(f'Arrive times: {arrive_times}')
        print(f'Score: {score}')
        print(f'Loss: {loss}')
        print('----------------------------------------------')

