import gymnasium
from gymnasium import spaces
import numpy as np

from DRL.Studienarbeiten.Studienarbeit2 import race
from race import *


class CurvyRaceEnv(gymnasium.Env):
    def __init__(self):
        super(CurvyRaceEnv, self).__init__()
        self.goals_hit = 0
        self.curr_step = 0
        # Initialize the CurvyRace environment
        self.curvy_race = CurvyRace()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None):
        # Reset the CurvyRace environment
        obs = self.curvy_race.reset()
        # Explicitly cast the observation to float32
        obs = np.array(obs, dtype=np.float32)
        info = {}  # You can provide additional information here if needed
        return obs, info

    def step(self, action):
        self.curr_step += 1
        # Take a step in the CurvyRace environment
        obs, reward, done = self.curvy_race.step(action)
        if reward == 1:
            self.goals_hit += 1
            reward = reward + self.goals_hit / self.curr_step
        else:
            reward = -0.1
        obs = np.array(obs, dtype=np.float32)

        return obs, reward, done, False, {}

    def render(self, mode='human'):
        # Render the CurvyRace environment (for visualization purposes)
        self.curvy_race.plot()

    def close(self):
        # Clean up resources, if any
        pass