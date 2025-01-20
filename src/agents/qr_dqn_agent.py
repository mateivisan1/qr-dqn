import torch
import random
import torch.optim as optim
from src.models.qr_dqn_model import QRCNN
from src.memory.replay_buffer import ReplayBuffer

class QRDQNAgent:
    def __init__(
        self,
        env,
        num_quantiles=51,
        gamma=0.99,
        lr=1e-4,
        batch_size=32,
        buffer_size=100000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.05,
        target_update_interval=10000,
        device=None
    ):
        self.env = env
        self.num_actions = env.action_space.n
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

        self.online_net = QRCNN(in_channels=1, num_actions=self.num_actions, num_quantiles=num_quantiles).to(self.device)
        self.target_net = QRCNN(in_channels=1, num_actions=self.num_actions, num_quantiles=num_quantiles).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.tau_hat = torch.linspace(0.0, 1.0, num_quantiles+1, device=self.device)[:-1] + 0.5 / num_quantiles

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # state shape: (84,84,1)
            # rearr => (1,1,84,84)
            state_t = torch.tensor(state, device=self.device).float().permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                quantiles = self.online_net(state_t)       # (1, num_actions, num_quantiles)
                q_values = quantiles.mean(dim=2)           # (1, num_actions)
                action = q_values.argmax(dim=1).item()
            return action

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self):
        """
        Sample from replay, do a single SGD step on the quantile huber loss.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device).float().permute(0,3,1,2)
        actions_t = torch.tensor(actions, device=self.device).long().unsqueeze(-1).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=self.device).float().unsqueeze(-1)
        next_states_t = torch.tensor(next_states, device=self.device).float().permute(0,3,1,2)
        dones_t = torch.tensor(dones, device=self.device).float().unsqueeze(-1)
        # print("rewards_t:", rewards_t.shape)
        # print("dones_t:", dones_t.shape)

        # current state-action quantiles
        quantiles_pred = self.online_net(states_t)
        # Gather the quantiles for the taken actions
        quantiles_pred_chosen = quantiles_pred.gather(
            dim=1,
            index=actions_t.expand(-1, -1, self.num_quantiles)
        )
        # (batch_size, 1, num_quantiles)
        quantiles_pred_chosen = quantiles_pred_chosen.squeeze(1)  # => (batch_size, num_quantiles)

        # next-state value
        with torch.no_grad():
            next_quantiles_online = self.online_net(next_states_t)
            #print("next_quantiles_online:", next_quantiles_online.shape) 
            next_q_online_mean = next_quantiles_online.mean(dim=2)
            #print("next_q_online_mean:", next_q_online_mean.shape)
            best_actions = next_q_online_mean.argmax(dim=1)
            #print("best_actions:", best_actions.shape)

            # b) Gather from target net
            next_quantiles_target = self.target_net(next_states_t)  # (batch_size, num_actions, num_quantiles)
            #print("next_quantiles_target:", next_quantiles_target.shape)
            # expand best_actions to shape (batch_size, 1, num_quantiles)
            best_actions_expanded = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_quantiles)
            #print("best_actions_expanded:", best_actions_expanded.shape)
            next_quantiles_target_chosen = next_quantiles_target.gather(
                dim=1,
                index=best_actions_expanded
            )  # (batch_size, 1, num_quantiles)
            #print("next_quantiles_target_chosen:", next_quantiles_target_chosen.shape)

            # bellman update
            #targets = rewards_t + (1.0 - dones_t) * self.gamma * next_quantiles_target_chosen
            # print("rewards_t:", rewards_t.shape)
            # print("dones_t:", dones_t.shape)
            # print("next_quantiles_target_chosen:", next_quantiles_target_chosen.shape)

            temp1 = (1.0 - dones_t)
            # print("temp1 shape:", temp1.shape)

            temp2 = temp1 * self.gamma
            # print("temp2 shape:", temp2.shape)
            temp2 = temp2.unsqueeze(-1)  # (32,1,1)

            temp3 = temp2 * next_quantiles_target_chosen
            # print("temp3 shape:", temp3.shape)

            rewards_t = rewards_t.unsqueeze(-1)  # (32,1,1)
            targets = rewards_t + temp3
            # print("targets (before squeeze):", targets.shape)
            # => (batch_size, 1, num_quantiles)
            # print("targets before squeeze:", targets.shape)
            targets = targets.squeeze(1)  # (batch_size, num_quantiles)
            # print("targets final:", targets.shape)

        # quantile Huber loss
        pairwise_delta = targets.unsqueeze(1) - quantiles_pred_chosen.unsqueeze(-1)
        # => shape: (batch_size, num_quantiles, num_quantiles)

        tau = self.tau_hat.view(1, -1, 1)  # shape (1, num_quantiles, 1)
        huber_loss = self.huber(pairwise_delta)  # (batch_size, num_quantiles, num_quantiles)

        indicator = (pairwise_delta.detach() < 0.0).float()
        quantile_weights = torch.abs(tau - indicator)
        loss = (quantile_weights * huber_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def huber(x, k=1.0):
        """
        Huber loss, threshold k=1.
        """
        cond = (x.abs() < k).float()
        return 0.5 * cond * x.pow(2) + (1 - cond) * k * (x.abs() - 0.5 * k)
