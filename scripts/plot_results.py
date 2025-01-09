import sys
import os

# Optionally ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

import json
import matplotlib.pyplot as plt

def plot_learning_curve(log_path, plot_path):
    with open(log_path, "r") as f:
        data = json.load(f)
    episodes = [d["episode"] for d in data]
    rewards = [d["reward"] for d in data]

    plt.figure(figsize=(10,6))
    plt.plot(episodes, rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("QR-DQN on Breakout - Learning Curve")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    plot_learning_curve("data/logs/training_log.json", "data/results/learning_curve.png")
