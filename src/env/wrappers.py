import gymnasium as gym
import cv2
import numpy as np

class PreprocessingWrapper(gym.ObservationWrapper):
    """
    A simple observation wrapper that keeps Atari frames in RGB
    but resizes them to 84x84 if desired.
    """
    def __init__(self, env):
        super().__init__(env)
        # We'll store frames as (84, 84, 3) in uint8
        # (You can adjust to whatever resolution you prefer)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

    def observation(self, obs):
        # obs has shape (210, 160, 3) for Atari
        # Resize to 84x84 while keeping 3 color channels
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # Now the shape is (84, 84, 3)
        return obs
