import gymnasium
from gymnasium import spaces
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
# from DRL.Studienarbeiten.Studienarbeit2 import race
from race import *


class CurvyRaceEnv(gymnasium.Env):
    def __init__(self):
        super(CurvyRaceEnv, self).__init__()

        # Initialize the CurvyRace environment
        self.curvy_race = CurvyRace()
        self.index_curr = 0

        # Setup action space and observation space
        low = np.array([-self.curvy_race.get_action_limits()[0], -self.curvy_race.get_action_limits()[1]])
        high = np.array([self.curvy_race.get_action_limits()[0], self.curvy_race.get_action_limits()[1]])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None):

        # Reset the CurvyRace environment
        obs = self.curvy_race.reset()
        self.index_curr = 0

        # add observations
        obs = self.__add_observations(obs)
        obs = np.array(obs, dtype=np.float32)

        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done = self.curvy_race.step(action)
        gate_passed = {'gate_passed': 0}

        # Made it to last Gate
        if self.curvy_race.get_gate_idx() == len(self.curvy_race.get_gates()):
            reward += 1000
            self.index_curr = 0
            gate_passed['gate_passed'] = 1
            observation = np.append(obs, [0, 0])
            observation = np.array(observation, dtype=np.float32)

            return observation, reward, done, False, gate_passed

        # add observations
        observation = self.__add_observations(obs)

        # calculate reward
        reward = -observation[4] / 14
        reward -= observation[3] / 2

        # passed gate
        if self.curvy_race.get_gate_idx() > self.index_curr:
            reward += 10 * self.curvy_race.get_gate_idx()
            gate_passed['gate_passed'] = 1
            self.index_curr += 1

        if done:
            self.index_curr = 0

        observation = np.array(observation, dtype=np.float32)

        return observation, reward, done, False, gate_passed

    def __add_observations(self, obs):
        """
        Augments observation data with additional features related to the agent's position and orientation relative to the next gate in a race track.

        Parameters:
        - obs (array-like): An array containing observation data, where:
            obs[0]: x-position of the agent
            obs[1]: y-position of the agent
            obs[2]: angle representing the orientation of the agent


        Returns:
        - obs (array-like): Augmented observation data including the following additional features:
            obs[5]: angle between the agent's orientation and the angle to the center of the next gate
            obs[6]: distance between the agent and the next gate
        """
        # Get x- and y-position of next gate center
        center_next_gate_x = self.curvy_race.get_gates()[self.curvy_race.get_gate_idx()][0][0]
        center_next_gate_y = (self.curvy_race.get_gates()[self.curvy_race.get_gate_idx()][1][1] +
                              self.curvy_race.get_gates()[self.curvy_race.get_gate_idx()][0][1]) / 2

        # Calulate angle difference between agent and next gate center
        obs = np.append(obs, self.__get_angle(center_next_gate_x, center_next_gate_y, obs))

        # Calculate distance between agent and next gate
        obs = np.append(obs, self.__dist_agent_gate(self.curvy_race.get_gates()[self.curvy_race.get_gate_idx()], obs))

        return obs

    def __get_angle(self, gate_x, gate_y, obs):
        """
        Calculates the angle by which the agent needs to turn to face the center of the next gate.

        Parameters:
        - gate_x (float): x-coordinate of the center of the next gate.
        - gate_y (float): y-coordinate of the center of the next gate.
        - obs (array-like): An array containing observation data, where:
            obs[0]: x-position of the agent
            obs[1]: y-position of the agent
            obs[2]: angle representing the orientation of the agent

        Returns:
        - turn_amount (float): The angle by which the agent needs to turn to face the center of the next gate, in radians.
        """
        # Calculate the vector from the agent to the gate
        vector_to_gate = np.array([gate_x - obs[0], gate_y - obs[1]])

        # Calculate the angle between the agent's forward direction and the vector to the gate
        angle_to_gate = np.arctan2(vector_to_gate[1], vector_to_gate[0])

        # Calculate the amount by which the agent needs to turn
        turn_amount = angle_to_gate - obs[2]

        # Ensure that the turn_amount is within the range [-pi, pi]
        turn_amount = (turn_amount + np.pi) % (2 * np.pi) - np.pi

        return turn_amount

    def __dist_agent_gate(self, gate, obs):
        """
        Calculates the Euclidean distance between the agent and the center of the next gate.

        Parameters:
        - gate (list): A list representing the coordinates of the next gate's endpoints, where:
            gate[0]: The coordinates of one endpoint of the gate.
            gate[1]: The coordinates of the other endpoint of the gate.
        - obs (array-like): An array containing observation data, where:
            obs[0]: x-position of the agent
            obs[1]: y-position of the agent

        Returns:
        - distance (float): The Euclidean distance between the agent and the center of the next gate.
        """

        gate_centerpoint = np.array([gate[0][0], (gate[0][1] + gate[1][1]) / 2])

        return math.sqrt((gate_centerpoint[0] - obs[0]) ** 2 + (gate_centerpoint[1] - obs[1]) ** 2)

    def render(self, mode='human'):
        # Render the CurvyRace environment (for visualization purposes)
        self.curvy_race.plot()

    def close(self):
        # Clean up resources, if any
        pass
