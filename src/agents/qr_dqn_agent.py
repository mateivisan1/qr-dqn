import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QRDQNAgent:
    def __init__(self, model, target_model, num_actions, num_quantiles=51, gamma=0.99, lr=1e-4):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Quantile fractions (tau_1..tau_N)
        self.tau = torch.linspace(0.0, 1.0, self.num_quantiles)
        self.tau = self.tau.view(1, self.num_quantiles)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        with torch.no_grad():
            state_t = torch.tensor(state).unsqueeze(0)
            quantiles = self.model(state_t)
            # Average across quantiles to get Q-values
            q_values = quantiles.mean(dim=2)
            action = q_values.argmax(dim=1).item()
        return action

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states)
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current dist
        dist = self.model(states)  # (B, A, Q)
        action_dist = dist.gather(
            1,
            actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_quantiles)
        ).squeeze(1)  # (B, Q)

        with torch.no_grad():
            # Next dist from target network
            next_dist = self.target_model(next_states)  # (B, A, Q)
            next_q_values = next_dist.mean(dim=2)  # (B, A)
            next_actions = next_q_values.argmax(dim=1)  # (B,)
            next_action_dist = next_dist.gather(
                1,
                next_actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_quantiles)
            ).squeeze(1)  # (B, Q)

            # Bellman update
            target = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_action_dist

        # Quantile Huber loss
        diff = target.unsqueeze(1) - action_dist.unsqueeze(2)  # (B, Q, Q)
        loss = self._quantile_huber_loss(diff, self.tau.to(diff.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _quantile_huber_loss(self, diff, tau):
        # diff shape: (B, Q, Q)
        huber_loss = torch.where(
            diff.abs() < 1.0,
            0.5 * diff.pow(2),
            diff.abs() - 0.5
        )
        # The quantile weighting factor
        weight = torch.abs(tau.unsqueeze(2) - (diff.detach() < 0).float())
        loss = (weight * huber_loss / self.num_quantiles).sum(dim=1).mean()
        return loss

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
