import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


class Minigrid:
    def __init__(self, env_name, view_size: int = 3, tile_size: int = 28, realtime_mode: bool = False):
        """Wrapper for MiniGrid environments with configurable view and tile sizes."""

        # Set the environment rendering mode
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"

        # Instantiate and wrap the environment
        self._env = gym.make(env_name, agent_view_size=view_size, tile_size=tile_size, render_mode=render_mode)
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=tile_size)
        self._env = ImgObsWrapper(self._env)

        # Derive observation space dynamically from a reset sample
        sample_obs, _ = self._env.reset(seed=np.random.randint(0, 99))
        sample_obs = sample_obs.astype(np.float32) / 255.0
        sample_obs = np.swapaxes(sample_obs, 0, 2)
        sample_obs = np.swapaxes(sample_obs, 2, 1)

        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return spaces.Discrete(3)

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset(seed=np.random.randint(0, 99))
        obs = obs.astype(np.float32) / 255.0
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)

        return obs

    def _format_action(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            return int(action[0])
        return action

    def step(self, action):
        action = self._format_action(action)
        obs, reward, done, truncated, _ = self._env.step(action)
        self._rewards.append(reward)
        obs = obs.astype(np.float32) / 255.0
        if done or truncated:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)

        return obs, reward, done or truncated, info

    def render(self):
        img = self._env.render()
        time.sleep(0.5)
        return img

    def close(self):
        self._env.close()