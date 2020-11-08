---
layout: default
title: Proposal
---

## Summary of the Project ##

Wall Climber is a struggle only for the most athletic of Steves. In Wall Climber, our agent is on a mission to climb to the top of a wall. Navigating through a maze of ladders, and by dodging arrows, our agent will rise to the top by learning where he can and cannot go. The input to the agent will be checkpoints that he reaches on the wall that reward him (this is to ensure that the agent doesn't simply learn to go straight up, as there may be cases where he needs to actually backtrack in order to go further). And the agent will lose points if he either falls off the wall, or gets hit by an arrow. The goal is to train our agent to be the fastest and most daring minecraft agent to scale any wall.  

## AI/ML Algorithms ##

The project will be using Reinforcement Learning (ie. Monte Carlo) and Greedy Algorithms (ie. Dijkstra’s or A*) to calculate the main goal of our project.

## Evaluation Plan ##

The reward, which will be our measuring metric, will be whenever the agent is able to reach a checkpoint block. Let's make this a diamond block. The reward will be +3 points, and a check point will be created every 5 layers he is able to go up. The penalties will be created by the arrows shooting at a specific block on the wall. These blocks will be labelled with obsidian blocks. The penalty will be -1 point every time he is hit by an arrow. The penalty will also be -10 points if he falls off the wall. This will be measured by the lava pit waiting for him at the bottom. These reward and penalty values will most likely be adjusted as a work in progress to figure out how to make the game balanced so that the agent is able to learn efficiently.

## Appointment with the Instructor ##

3:00pm Oct. 22, 2020
