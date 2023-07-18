#!/usr/bin/env python3
import numpy as np
import helpersContinuous
from ContinuousGridworld import *

######## Defining gridworld specs

length = 1.0
stepsize = 0.2
noise = 0.1

## irrelevant but required
discount = 0.9
rewards = np.array([[0.8, 0.8], [1.0, 1.0]])

######## Defining Environment Storage


maxSteps = 30

def worldEnv(s, discretization):

 world = ContinuousGridworld(length=length, stepsize=stepsize, discretization=discretization, noise=noise, discount=discount, rewards=rewards)
 # Initializing environment
 
 world.setRandomPosition()
 s["State"] = world.getPosition().tolist()
 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  action = s["Action"]
  done = world.move(action[0])
  #print(s["Action"]) 
  
  # Getting Reward
  s["Reward"] = world.getReward()
  
  # Storing New State
  state = world.getPosition().tolist()
  s["State"] = state
  
  # Getting Features
  #print(state)
  features = helpersContinuous.getGaussianWeightFeatures(world, state)
  s["Features"] = features
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 s["Termination"] = "Terminal"
