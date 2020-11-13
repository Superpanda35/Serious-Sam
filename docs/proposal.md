---
layout: default
title: Proposal
---

## Summary of the Project ##

Our hero Sam is fighting zombies in a enclosed arena in Minecraft. Sam should detect zombies who approching to him, and properly find best angle to attack the nearest enemy from all directions while he moving to avoid zombies' attack. The main goal is to kill all the zombies and survival. The optimal way is not only survival, but also to get as less damage as possible in the limited time.  

## AI/ML Algorithms ##

The project will be using Reinforcement Learning (ie. Monte Carlo) and Greedy Algorithms (ie. Dijkstra's or A*) to calculate and reach the main goal of our project.

## Evaluation Plan ##

The reward, which will be our measuring metric, will be whenever the agent is to kill and damage the zombies. And the penalties will be created by got damaged and killed from the zombies. For example, killing the zombies will be +10, and damage to the zombies will be +5. Killed by zombies will be -20, and damaged by zombies will be -8. In this calculating, the agent Sam will be trained hard for getting a good score. These reward and penalty values will most likely be adjusted as a work in progress to figure out how to make the game balanced so that the agent is able to learn efficiently.

## Appointment with the Instructor ##

3:00pm Oct. 22, 2020
