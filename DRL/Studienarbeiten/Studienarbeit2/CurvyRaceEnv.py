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
        self.last_action = []
        self.dist_to_gate = self.curvy_race.get_gates()[0][0][0]
        self.steps_since_last_goal = 0
        low = np.array([-self.curvy_race.get_action_limits()[0], -self.curvy_race.get_action_limits()[1]])
        high = np.array([self.curvy_race.get_action_limits()[0], self.curvy_race.get_action_limits()[1]])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None):
        # Reset the CurvyRace environment
        obs = self.curvy_race.reset()
        obs = np.append(obs, self.dist_agent_gate(self.curvy_race.get_gates()[self.curvy_race.gate_idx], obs))
        obs = np.append(obs, self.steps_since_last_goal)
        # Explicitly cast the observation to float32
        obs = np.array(obs, dtype=np.float32)
        info = {}  # You can provide additional information here if needed
        return obs, info

    def step(self, action):
        self.curr_step += 1

        # Take a step in the CurvyRace environment
        obs, reward, done = self.curvy_race.step(action)
        if reward == 1:
            self.steps_since_last_goal = 0
            
            #print("goal")
            self.goals_hit += 1
            reward = reward + 5*self.goals_hit
            self.dist_to_gate = self.dist_agent_gate(self.curvy_race.get_gates()[self.curvy_race.gate_idx], obs)
            obs = np.append(obs, self.dist_to_gate)
            obs = np.append(obs, self.steps_since_last_goal)
        else:
            self.steps_since_last_goal += 1
            new_dist_to_gate = self.dist_agent_gate(self.curvy_race.get_gates()[self.curvy_race.gate_idx], obs)
            reward = -self.steps_since_last_goal    #always minus ten

            #moving away from gate
            if new_dist_to_gate > self.dist_to_gate:
                reward -= 10
            
            #moving towards gate significantly
            if abs(new_dist_to_gate - self.dist_to_gate) >= 0.5:
                reward = 5 - self.steps_since_last_goal
           
            obs = np.append(obs, new_dist_to_gate)
            obs= np.append(obs, self.steps_since_last_goal)

            


            self.dist_to_gate = new_dist_to_gate
           
        obs = np.array(obs, dtype=np.float32)

        return obs, reward, done, False, {}

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
