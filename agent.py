import json
import time
from collections import defaultdict, deque
import random
import sys
import math
import utils
rom tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Agent:

    #agent constructor
    def __init__(self,alpha, gamma,epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = dict()
        self.pastActions = []
        self.possibleActions = {}

    # get observations from world state, returns a world state dictionary
    def getObservations(self,world_state) -> dict:
        observations = {}
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

        return observations

    def chooseActions(self,q_network,epsilon,action_values):

        """
        Select action according to e-greedy policy

        Args:
            action_values: list of possible actions
            q_network (QNetwork): Q-Network
            epsilon (float): probability of choosing a random action

        Returns:
            action (int): chosen action [0, action_size)
        """

        # Prevent computation graph from being calculated
        with torch.no_grad():
            # Calculate Q-values fot each action
            obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
            action_values = q_network(obs_torch)

            # e_greedy = np.random.random()
            e_greedy = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
            if e_greedy:
                # e-greedy take a random action
                action_idx = randint(0, len(action_values[0]))
            else:
                # Select action with highest Q-value
                action_idx = torch.argmax(action_values).item()

        return action_idx











