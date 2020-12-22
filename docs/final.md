---
layout: default
title: Final Report
---

## Video ##


## Project Summary ##

Our Project is Serious-Sam, based on the game, has a arena of size 8 by 8 and has to kill a zombie within time limits. The goal of the project is to teach the agent, which we named Serious Sam, to kill the Zombie without damaging itself and the evnironment outside. We need to use AI/ML for this project because there are many strategies and features such as the location, angle, and surrounding area that affects the agent's performance, similar to when you are in a fighting club. Therefore, we decided to use deep q learning to achieve our goal for this project. 

From the status, we have updated the size of our arena, the reward function, the number of Zombies and state space. For the final submission, we were able to improve upon the agent's performance. Before our agent was lucky a few times, but mainly was losing it's health while never even damaging the Zombie. We learned our mistake was the way we were calculating the observation space. It wasn't trivial to include the arena's wall location into the observation for the agent, because we assumed that the agent didn't need that information in order to kill the Zombie, however, later we learned that it is important because if the agent is facing the wall it cannot kill the Zombie. We also changed the reward function as we were learning that the reward functrion is very important to how the agent's performance is. A different reward function changed our agent's actions and events completely. We also learned that since killing a Zombie takes more than one hit, we needed to have the agent explore as much as it can to eventually kill the Zombie. For the final project, compared to the status project, the goal was to have the agent actually learn how to kill the zombie and keep improving with each step it takes. 


## Approach ##

We continued from Status with the deep q learning neural network approach for the agent to learn. We decided on deep q learning network because compared to using just q learning by itself, deep q learning will be able to take a much bigger observation and give you a result much faster. Since our observation was of size (9,5,5) due to the zombie's height, we needed to make sure that our model could handle a big observation. Though the equation we were using was the exact same to any q learning algorithm. We started with epsilon, our exploration rate, at 1, therefore the agent was exploring actions and states for majority of the beginning episodes. After that we had a epsilon decay of 0.99, so the agent start to learn by taking the actions that give the highest Q-value. Since initially the agent has no clue what action will lead to what result or reward, it was important to give the agent time to see the possibilities. However the main changes that we made were in the reward function. Since in the Status, the agent would keep dying and for each death, the agent received the same reward, it didn't learn to stay alive and try to kill the Zombie.


### Approach 1: Adding a positive timestep reward ###

First, we tried to do the opposite. We added a small positive reward for each time step the agent was alive, to encourage it to stay alive longer so it could have a higher chance of killing the Zombie. However this resulted in the opposite, instead the agent would learn to kill the Zombie at the end of the episode instead of the beginning since it was getting more reward from that. It also, sometimes would not kill the Zombie and be satisfied with the positive reward it accumulated just for being alive throughout the episode. 


### Approach 2: Adding a negative timestep reward ###

Next, we tried to add a small negative reward for each step the agent was alive, in hopes to encourage the agent to kill the zombie faster. However, it did the opposite. The agent instead was more rushed to finish the episode and die quicker as that would lead to less negative reward. 

## Evaluation ##





## References ##
build_test.py, mob_fun.py, tabular_q_learning.py and moving_target_test.py from Canvas page. <br />
https://eg.bucknell.edu/~cld028/courses/379-FA19/MalmoDocs/classmalmo_1_1_mission_spec-members.html <br />
https://microsoft.github.io/malmo/0.17.0/Python_Examples/Tutorial.pdf <br />
https://tsmatz.wordpress.com/2020/07/09/minerl-and-malmo-reinforcement-learning-in-minecraft/ <br />
https://github.com/microsoft/malmo/blob/master/Schemas/Types.xsd <br />
https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html#element_Weather <br />
https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
