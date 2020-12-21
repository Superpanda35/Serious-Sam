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
from agent import Agent
from world import World


# Hyperparameters
SIZE = 3 #half the width of the fighting box
OBS_SIZE = 3
MAX_EPISODE_STEPS = 80
MAX_GLOBAL_STEPS = 50000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1
ZOMBIES = 1
ACTION_DICT = {
    0: 'move 1',  # Move one block forward
    1: 'turn 1',  # Turn 90 degrees to the right
    2: 'turn -1',  # Turn 90 degrees to the left
    3: 'attack 1'  # Destroy block
}


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




def main():


    # create the DQN agent
    epsilon = 1

    agent = Agent(alpha=LEARNING_RATE, gamma = GAMMA, epsilon = epsilon,
                  input_dims = (OBS_SIZE,OBS_SIZE),batch_size = BATCH_SIZE,
                  n_actions = len(ACTION_DICT), max_mem_size = REPLAY_BUFFER_SIZE, eps_end = MIN_EPSILON,
                  eps_dec = EPSILON_DECAY)



    #create the Malmo world environment
    world = World(size = SIZE, obs_size = OBS_SIZE, num_entities = ZOMBIES, episodes = MAX_EPISODE_STEPS)

    # Init vars
    global_step = 0
    num_episode = 0
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
        agent_host = world.init_malmo()
        world_state = world.get_world_state()

        while not world_state.has_mission_begun:

            time.sleep(0.1)
            world_state = world.get_world_state()
            for error in world_state.errors:
                print("\nError:", error.text)

        #get the first observation
        obs , zombies_killed, health = world.get_observation()
        death = False
        total_killed = 0

        # Run the episode
        while world.is_mission_running:
            # print(".", end="")
            # time.sleep(0.1)

            # Get action
            #need to add a check that the agent doesn't attack the wall
            action_idx = agent.choose_action(obs)
            command = ACTION_DICT[action_idx]
            #print("action taken", command)

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(0.1)

            episode_step += 1

            #three terminal states
            # 1. if all the zombies have been killed
            # 2. the agent has lost all its health points and has died
            # 3. episode has take maximum steps
            if (episode_step > 10 and total_killed==ZOMBIES) or health == 0 or episode_step >= MAX_EPISODE_STEPS:
                done = True

                if episode_step >= MAX_EPISODE_STEPS or ZOMBIES-total_killed == 0:
                    print("sending the quit command", "zombies killed", total_killed, "reward", episode_return, "episode step", episode_step)
                    agent_host.sendCommand("quit")
                    #since quit makes health 0, change it to any number so the reward is calculated properly
                time.sleep(2)

            #get the agent's health value
            if health == 0:
                death = True

            # Get next observation
            world_state = world.get_world_state()

            for error in world_state.errors:
                print("Error:", error.text)


            next_obs, zombies_killed, health= world.get_observation()
            total_killed += zombies_killed

            # Get reward
            reward = world.get_reward(death, zombies_killed , episode_steps = episode_step)
            episode_return += reward
            #print("reward", reward)

            # Store step in replay buffer
            agent.store_transition(obs,action_idx,reward, next_obs,done)
            obs = next_obs


            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                loss = agent.learn()
                if loss == None:
                    loss = 0
                episode_loss += loss




        num_episode += 1
        zombies_killed  = 0
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







if __name__ == "__main__":
    main()