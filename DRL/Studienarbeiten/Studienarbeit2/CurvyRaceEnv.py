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

        self.move_since_last_gate = 0
        low = np.array([-self.curvy_race.get_action_limits()[0], -self.curvy_race.get_action_limits()[1]])
        high = np.array([self.curvy_race.get_action_limits()[0], self.curvy_race.get_action_limits()[1]])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=float)

        # obs_low = np.array([-5, -10, - self.curvy_race.get_action_limits()[1]])
        # obs_high = np.array([40, 10, self.curvy_race.get_action_limits()[1]])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None):
        
        # Reset the CurvyRace environment
        obs = self.curvy_race.reset()
        self.prev_pos = [0,0]
        # Explicitly cast the observation to float32
        obs = self.__add_observations(obs)
        self.prev_gate_dist = obs[3]
        obs = np.array(obs, dtype=np.float32)

        info = {}  # You can provide additional information here if needed
        return obs, info

    def step(self, action):        
        # Take a step in the CurvyRace environment
        obs, reward, done = self.curvy_race.step(action)
        obs = self.__add_observations(obs)
        if done and self.curvy_race.get_gate_idx() == 0:
            obs = np.array(obs, dtype=np.float32)
            reward = 200
            return obs, reward, done, False, {}

        if reward == 1:
            self.move_since_last_gate = 0
            reward = reward * 10 *self.curvy_race.get_gate_idx()


            self.next_goal = self.curvy_race.get_gates()[self.curvy_race.get_gate_idx()]
            self.prev_pos = [obs[0], obs[1]]
            self.prev_gate_dist = obs[3]
        else:
            self.move_since_last_gate +=1

            # reward based on distance to next gate
            reward = self.curvy_race.get_gate_idx()*(2 - obs[3])

            #MOVE FFS!!
            if self.prev_pos[0] == obs[0] and self.prev_pos[1] == obs[1]:
                reward = -20

            # distance to next gate to big
            if obs[3] >= 2:
                reward = -20

            # got closer/farer to gate
            if self.prev_gate_dist <= obs[3]:
                reward -= 5
            else:
                reward += 5

            reward -= self.move_since_last_gate

            self.prev_pos = [obs[0], obs[1]]
            self.prev_gate_dist = obs[3]


        #reward for being closer to last gate
        reward += 1.5*(32 - obs[4])
        obs = np.array(obs, dtype=np.float32)

        return obs, reward, done, False, {}



    

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

    def get_angle(self, gate, obs):
        #angle_bounds = []
        #angle_bounds.append(math.sin(self.dist_to_gate / abs(gate[1][1] - obs[1])))
        #angle_bounds.append(math.sin(abs(gate[1][0] - obs[1] / self.dist_to_gate)))

        return math.sin(self.dist_agent_gate(self.next_goal, obs)/abs((gate[1][1] - 1) - obs[1]))

    def dist_agent_gate(self, gate, obs):
        gate_centerpoint = np.array([gate[0][0], (gate[0][1] + gate[1][1])/2])

        return np.linalg.norm(gate_centerpoint - np.array([obs[0], obs[1]]))

    def render(self, mode='human'):
        # Render the CurvyRace environment (for visualization purposes)
        self.curvy_race.plot()

    def close(self):
        # Clean up resources, if any
        pass
