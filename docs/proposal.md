---
layout: default
title: Proposal
---

# Proposal: MineEats #

## Summary of Project ##

In light of the COVID-19 situation in the real world, and the rising popularity of food delivery services, this project (MineEats) will simulate food delivery in a city by using an agent to deliver to customers in different locations of the city. The agent will decide which orders should be delivered first based on the time it takes and the distance to the destination. The input to the agent will be the number and location of the customers, their order times, number of orders, and a restaurant location (to figure out the distance to pick up the order and distance to deliver the order). The goal of this project is to find the best way to deliver food in the shortest time and distance.

## AI/ML Algorithms ##

The project will be using Reinforcement Learning (ie. Monte Carlo) and Greedy Algorithms (ie. Dijkstra’s or A*) to calculate the main goal of our project.

## Evaluation Plan ##

The reward, which will be our measuring metric, for the agent will be based on the time it took and the distance the delivery man had to travel. Each time the user has to travel a mile long or wastes an extra minute the agent will be rewarded with a negative reward that way the agent eventually learns that it needs to find the optimal path. We can then evaluate whether the agent has been properly trained or not by giving it edge cases, and possible odd cases where orders overlap, or orders are coming/going to & from either the same restaurant or customer. Furthermore, we can analyze with our own judgment and knowledge of basic geometry/triangles, whether the agent is indeed taking the optimal route or not.
 
## Appointment with the Instructor ##

3:00pm Oct. 22, 2020
