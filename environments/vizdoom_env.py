import numpy as np
from gymnasium import spaces
from vizdoom import DoomGame, ScreenFormat, ScreenResolution


class VizdoomEnv:
    def __init__(self, config_path: str, scenario_path: str | None = None,
                 frame_skip: int = 4, resolution: str = "RES_160X120",
                 realtime_mode: bool = False) -> None:
        """Minimal VizDoom wrapper normalising observations and exposing Gym interface."""

        self._game = DoomGame()
        self._game.load_config(config_path)
        if scenario_path:
            self._game.set_doom_scenario_path(scenario_path)
        self._game.set_screen_format(ScreenFormat.RGB24)
        self._game.set_screen_resolution(getattr(ScreenResolution, resolution))
        self._game.set_window_visible(realtime_mode)
        self._game.init()

        self._frame_skip = frame_skip
        self._realtime_mode = realtime_mode

        shape = (3, self._game.get_screen_height(), self._game.get_screen_width())
        self._observation_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        n_buttons = self._game.get_available_buttons_size()
        self._action_space = spaces.MultiBinary(n_buttons)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_obs(self):
        state = self._game.get_state()
        screen = np.zeros(self._observation_space.shape, dtype=np.float32)
        if state is not None:
            screen = state.screen_buffer.astype(np.float32) / 255.0
            screen = np.transpose(screen, (2, 0, 1))
        return screen

    def reset(self):
        self._game.new_episode()
        return self._get_obs()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        reward = self._game.make_action(action, self._frame_skip)
        done = self._game.is_episode_finished()
        obs = self._get_obs()
        if done:
            info = {
                "reward": self._game.get_total_reward(),
                "length": self._game.get_episode_time(),
            }
        else:
            info = None
        return obs, reward, done, info

    def render(self):
        if self._realtime_mode:
            return self._game.get_state().screen_buffer
        return self._get_obs()

    def close(self):
        self._game.close()
