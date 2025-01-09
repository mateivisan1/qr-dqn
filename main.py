#!/usr/bin/env python3
import sys
import os
import json
import torch
import gymnasium as gym
import ale_py

# 1. Make sure Python knows where to find "src/"
#    We assume this file is inside a folder like "scripts/", so we go one level up.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.join(current_dir, '..')  # go up one directory
# src_path = os.path.join(project_root, 'src')
# if src_path not in sys.path:
#     sys.path.append(src_path)

# 2. Now we can import from src
from src.models.qr_dqn_model import QRDQNModel
from src.agents.qr_dqn_agent import QRDQNAgent
from src.memory.replay_buffer import ReplayBuffer


def main():
    """
    This main() function performs training only. 
    You can adapt this file later to do evaluation by modifying the code accordingly.
    """
    # A. Create the Breakout environment
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    print("Environment generated!!!")

    # B. Prepare model and agent
    #    - If you're using raw color frames (210x160x3), set in_channels=3
    #    - If using a grayscale wrapper, set in_channels=1
    model = QRDQNModel(num_actions=env.action_space.n)
    target_model = QRDQNModel(num_actions=env.action_space.n)
    target_model.load_state_dict(model.state_dict())
    print("models created!!!")

    agent = QRDQNAgent(
        model=model,
        target_model=target_model,
        num_actions=env.action_space.n,
        num_quantiles=51,
        gamma=0.99,
        lr=1e-4
    )
    print("agent created!!!")

    # C. Replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)
    print("replay buffer!!!")

    # D. Training hyperparameters
    episodes = 500
    batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 50000  # steps
    epsilon = epsilon_start

    log_data = []
    global_step = 0

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Epsilon decays linearly over 'epsilon_decay' steps
            epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end)/epsilon_decay)

            # Select action
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            global_step += 1

            # Update the agent
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)

            # Periodically update target network
            if global_step % 1000 == 0:
                agent.update_target()

        log_data.append({"episode": ep, "reward": episode_reward})
        print(f"Episode {ep}: Reward = {episode_reward:.2f}, Epsilon = {epsilon:.3f}")

        # Save intermediate checkpoints
        if (ep + 1) % 50 == 0:
            os.makedirs("data/checkpoints", exist_ok=True)
            torch.save(agent.model.state_dict(), f"data/checkpoints/model_ep{ep+1}.pth")

    # E. Final save
    os.makedirs("data/checkpoints", exist_ok=True)
    torch.save(agent.model.state_dict(), "data/checkpoints/model_final.pth")

    os.makedirs("data/logs", exist_ok=True)
    with open("data/logs/training_log.json", "w") as f:
        json.dump(log_data, f)

    print("Training finished. Model saved to data/checkpoints/model_final.pth")


if __name__ == "__main__":
    main()
