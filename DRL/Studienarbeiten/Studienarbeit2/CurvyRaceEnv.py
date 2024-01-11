import gymnasium
from gymnasium import spaces
import numpy as np
import math
# from DRL.Studienarbeiten.Studienarbeit2 import race
from race import *


class CurvyRaceEnv(gymnasium.Env):
    def __init__(self):
        super(CurvyRaceEnv, self).__init__()
        self.goals_hit = 0
        self.curr_step = 0
        # Initialize the CurvyRace environment
        self.curvy_race = CurvyRace()
        low = np.array([-self.curvy_race.get_action_limits()[0], -self.curvy_race.get_action_limits()[1]])
        high = np.array([self.curvy_race.get_action_limits()[0], self.curvy_race.get_action_limits()[1]])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=float)

        # obs_low = np.array([-5, -10, - self.curvy_race.get_action_limits()[1]])
        # obs_high = np.array([40, 10, self.curvy_race.get_action_limits()[1]])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None):
        # Reset the CurvyRace environment
        obs = self.curvy_race.reset()
        # Explicitly cast the observation to float32
        #obs = self.__add_observations(obs)
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



    def __calculate_gate_reward(self):
        self.gates_hit += 1
        return 2 * self.gates_hit

    def __calculate_angle_reward(self, angle, upperbound, lowerbound):
        if lowerbound <= angle <= upperbound:
            return 2
        else:
            return -2

    def __add_observations(self, obs):
        """
        obs[0]: x-position
        obs[1]: y-position
        obs[2]: winkel
        obs[3]: distanz zum nächsten gate
        obs[4]: nötiger winkel lower bound
        obs[5]: nötiger winkel upper bound
        obs[6]: Count gates hit
        """
        obs = np.append(obs, self.dist_agent_gate(self.curvy_race.get_gates()[self.curvy_race.gate_idx], obs))
        obs = np.append(obs, self.get_angle(self.curvy_race.get_gates()[self.curvy_race.gate_idx], obs))
        obs = np.append(obs, self.gates_hit)
        return obs

    def get_angle(self, gate, obs):
        angle_bounds = []
        angle_bounds.append(math.sin(self.dist_to_gate / abs(gate[1][1] - obs[1])))
        angle_bounds.append(math.sin(abs(gate[1][0] - obs[1] / self.dist_to_gate)))

        return angle_bounds

    def dist_agent_gate(self, gate, obs):
        # V: vector representing the line segment
        V = np.array([gate[1][1] - gate[0][0], gate[1][0] - gate[0][1]])

        # W: vector representing the line segment from P1 to Q
        W = np.array([obs[0] - gate[0][0], obs[1] - gate[0][1]])

        # Projection of W onto V
        projection = np.dot(W, V) / np.dot(V, V) * V

        # Point P' on the line segment
        P_prime = np.array([gate[0][0], gate[1][1]]) + projection

        # Distance between Q and P'
        distance = np.linalg.norm(np.array([obs[0], obs[1]]) - P_prime)

        return distance

    def render(self, mode='human'):
        # Render the CurvyRace environment (for visualization purposes)
        self.curvy_race.plot()

    def close(self):
        # Clean up resources, if any
        pass
