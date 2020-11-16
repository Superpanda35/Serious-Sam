from __future__ import print_function

from builtins import range
from malmo import MalmoPython
import os
import sys
import time
import json
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import random
import math

# Hyperparameters
SIZE = 50
REWARD_DENSITY = .1
PENALTY_DENSITY = .02
OBS_SIZE = 5
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1
ACTION_DICT = {
    0: 'move 1',  # Move one block forward
    1: 'turn 1',  # Turn 90 degrees to the right
    2: 'turn -1',  # Turn 90 degrees to the left
    3: 'attack 1'  # Destroy block
}


# Q-Value Network
class QNetwork(nn.Module):

    def __init__(self, obs_size, action_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size))

    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately from agent_visibility_test.py
else:
    import functools

    print = functools.partial(print, flush=True)


def drawWall(blocktype, height):
    genString = ""

    for i in range(1, height + 1):
        genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
            i) + '" x1="-16" x2="16" z1="-16" z2="-16" />'
        genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
            i) + '" x1="-16" x2="-16" z1="-16" z2="16" />'
        genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
            i) + '" x1="16" x2="16" z1="-16" z2="16" />'
        genString += '<DrawLine type="' + blocktype + '" y1="' + str(i) + '" y2="' + str(
            i) + '" x1="-16" x2="16" z1="16" z2="16" />'

    return genString


def GetMissionXML():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Kill Zombies</Summary>
              </About>

              <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>9000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                <AllowSpawning> false </AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,2;1;"/>
                    <DrawingDecorator>
                        ''' + drawWall("cobblestone_wall", 10) + '''

                        <DrawEntity x="0" y="7" z="3" type="Zombie"/>
                        <DrawEntity x="2" y="7" z="5" type="Zombie"/>
                        <DrawEntity x="-5" y="7" z="10" type="Zombie"/>
                        <DrawEntity x="10" y="7" z="0" type="Zombie"/>
                    </DrawingDecorator>

                  <ServerQuitFromTimeUp timeLimitMs="15000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Serious Sam</Name>
                <AgentStart>
                    <Placement x="0" y="7" z="0"/>
                    <Inventory>
                        <InventoryItem slot="0" type="golden_sword"/>
                    </Inventory>
                </AgentStart>

                <AgentHandlers>
                    <RewardForDamagingEntity>
                        <Mob type="Zombie" reward="100"/>
                    </RewardForDamagingEntity>
                    <RewardForMissionEnd rewardForDeath="-1000">
                        <Reward description ="out_of_time" reward="-100"/>
                    </RewardForMissionEnd>
                    <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-''' + str(int(OBS_SIZE / 2)) + '''" y="-1" z="-''' + str(int(OBS_SIZE / 2)) + '''"/>
                                <max x="''' + str(int(OBS_SIZE / 2)) + '''" y="0" z="''' + str(int(OBS_SIZE / 2)) + '''"/>
                            </Grid>
                        </ObservationFromGrid>
                    <ObservationFromRay/>
                    <ObservationFromNearbyEntities>
                        <Range name="entities" xrange="100" yrange="2" zrange="100" update_frequency="1"/>
                    </ObservationFromNearbyEntities>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="120"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


def get_action(obs, q_network, epsilon):
    """
    Select action according to e-greedy policy

    Args:
        obs (np-array): current observation, size (obs_size)
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


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "DiamondCollector")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


def get_observation(world_state):
    """
    Use the agent observation API to get a 2 x 5 x 5 grid around the agent.
    The agent is in the center square facing up.

    Args
        world_state: <object> current agent world state

    Returns
        observation: json object
    """
    obs = np.zeros((2, OBS_SIZE, OBS_SIZE))
    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            print("observations full stats", observations['DamageTaken'], observations['Life'])


            # Get observation
            if "floorAll" in observations:
                grid = observations['floorAll']
                grid_binary = [1 if x == "cobblestone_wall" else 0 for x in grid]
                obs = np.reshape(grid_binary, (2, OBS_SIZE, OBS_SIZE))

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

            # return observations

        return obs


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)

    return obs, action, next_obs, reward, done


def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Zombie Killer')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value))


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:

            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)
        obs = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running:
            # Get action
            # print("observation", obs)
            action_idx = get_action(obs, q_network, epsilon)
            command = ACTION_DICT[action_idx]
            print("action taken", command)
            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(2)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            # need to add the part where the episode ends if all the zombies
            if episode_step >= MAX_EPISODE_STEPS or \
                    (obs[0, int(OBS_SIZE / 2) - 1, int(OBS_SIZE / 2)] == 1 and \
                     obs[1, int(OBS_SIZE / 2) - 1, int(OBS_SIZE / 2)] == 0 and \
                     command == 'move 1'):
                done = True
                time.sleep(2)

                # Get next observation
            world_state = agent_host.getWorldState()

            for error in world_state.errors:
                print("Error:", error.text)
                print("HEREEEEEE")
                break
            next_obs = get_observation(world_state)

            # Get reward
            reward = 0
            for r in world_state.rewards:
                add = r.getValue()
                if add == 100:
                    print("hit zombie")
                reward += add
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        # print("final observations", get_observation(world_state), "return", episode_return)
        print("final world state", world_state.is_mission_running)
        num_episode += 1
        # hard coding the issue right now
        # if episode_return == 0:
        #     print("bad run")
        #     episode_return =-1000
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description(
            'Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
                num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, returns)
            print()


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)




