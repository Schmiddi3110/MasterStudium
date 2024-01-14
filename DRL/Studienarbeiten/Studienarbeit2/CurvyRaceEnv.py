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
        self.prev_pos = [0,0]
        self.next_goal = self.curvy_race.get_gates()[self.curvy_race.get_gate_idx()]
        self.prev_gate_dist = 2
        self.index_curr = 0
        self.move_since_last_gate = 0
        low = np.array([-self.curvy_race.get_action_limits()[0], -self.curvy_race.get_action_limits()[1]])
        high = np.array([self.curvy_race.get_action_limits()[0], self.curvy_race.get_action_limits()[1]])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        # obs_low = np.array([-5, -10, - self.curvy_race.get_action_limits()[1]])
        # obs_high = np.array([40, 10, self.curvy_race.get_action_limits()[1]])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None):
        
        # Reset the CurvyRace environment
        obs = self.curvy_race.reset()
        self.prev_pos = [0,0]
        self.index_curr  =0
        # Explicitly cast the observation to float32
        #obs = self.__add_observations(obs)
        #self.prev_gate_dist = obs[3]
        obs = np.append(obs, [0,0])
        obs = np.array(obs, dtype=np.float32)

        info = {}  # You can provide additional information here if needed
        return obs, info

    def step(self, action):        
        obs, reward, done = self.curvy_race.step(action)

        gates = self.curvy_race.get_gates()
        index_next = self.curvy_race.get_gate_idx()

        if index_next == 16:
            reward += 1000
            self.index_curr = 0

            observation = np.append(obs, [0,0])
            observation = np.array(observation, dtype=np.float32)
            return observation, reward, done, False, {}

        next_gate = gates[index_next]
        center_next_gate_x = next_gate[0][0]
        center_next_gate_y = (next_gate[1][1] + next_gate[0][1])/2

        angle_to_next_gate = self.get_angle(center_next_gate_x, center_next_gate_y, obs)

        gatedist = self.dist_agent_gate(next_gate, obs)

        observation = np.append(obs, [angle_to_next_gate, gatedist])

        reward = -gatedist/14
        reward -= angle_to_next_gate/2

        if index_next > self.index_curr:
            reward += 10*self.curvy_race.get_gate_idx()

            self.index_curr += 1

        if done:
            self.index_curr = 0


        observation = np.array(observation, dtype=np.float32)

        return observation, reward, done, False, {}



    

    def __add_observations(self, obs):
        """
        obs[0]: x-position
        obs[1]: y-position
        obs[2]: winkel
        obs[3]: distanz zum nächsten gate
        obs[4]: distanz zum letzten tor
        """
        obs = np.append(obs, self.dist_agent_gate(self.next_goal, obs))
        obs = np.append(obs, self.dist_agent_gate(self.curvy_race.get_gates()[self.curvy_race.get_gates().__len__()-1], obs))


        
        return obs

    def get_angle(self, gate_x, gate_y, obs):
        # Calculate the vector from the agent to the gate
        vector_to_gate = np.array([gate_x - obs[0], gate_y - obs[1]])

        # Calculate the angle between the agent's forward direction and the vector to the gate
        angle_to_gate = np.arctan2(vector_to_gate[1], vector_to_gate[0])

        # Calculate the amount by which the agent needs to turn
        turn_amount = angle_to_gate - obs[2]

        # Ensure that the turn_amount is within the range [-pi, pi]
        turn_amount = (turn_amount + np.pi) % (2 * np.pi) - np.pi

        return turn_amount


    def dist_agent_gate(self, gate, obs):
        gate_centerpoint = np.array([gate[0][0], (gate[0][1] + gate[1][1])/2])

        return math.sqrt((gate_centerpoint[0] - obs[0])**2 + (gate_centerpoint[1] - obs[1])**2)

    def render(self, mode='human'):
        # Render the CurvyRace environment (for visualization purposes)
        self.curvy_race.plot()

    def close(self):
        # Clean up resources, if any
        pass
