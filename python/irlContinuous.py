from ContinuousGridworld import *
from GridWorldEnv import *
import helpersContinuous

from scipy.optimize import linprog
import numpy as np
import argparse
import json
import korali

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1, help='number irl iterations')
    parser.add_argument('--discount', type=float, default=0.90, help='discount factor')
    parser.add_argument('--noise', type=float, default=0.1, help='action noise')
    parser.add_argument('--epsilon', type=float, default=0.01, help='accuracy of value iteration')
    parser.add_argument('--discretization', type=int, default=5, help='action noise')
    parser.add_argument('--numobs', type=int, default=1, help='number observed expert trajectories')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--noisy', action='store_true', help='print output from value iteration')

    ## Parse arguments
    
    args = parser.parse_args()

    N = args.discretization
    p = args.noise
    gamma = args.discount
    maxiterations = args.iteration
    numobs = args.numobs
    epsilon = args.epsilon
    noisy = args.noisy
    obsfile = 'observations.json'

    ## Initialization

    np.random.seed(args.seed)

    # create reward quadrant

    rewards = np.array([[0.8, 0.8], [1.0, 1.0]])
    
    # find optimal policy

    world = ContinuousGridworld(length=1.0, stepsize=0.2, discretization=N, noise=p, discount=gamma, rewards=rewards)
    valueMatrix, policyMatrix = helpersContinuous.doDiscretizedValueIteration(world, epsilon, 1e4, noisy=noisy)
 
    print(valueMatrix)
    print(policyMatrix)

    allStates = []
    allActions = []
    allFeatures = []
    states, actions = helpersContinuous.doRollout(world, policyMatrix, 30)

    allStates.append(states[:-1])
    allActions.append(actions)

    #print(states)
    #print(actions)

    sumGaussianWeights = helpersContinuous.calculateGaussianWeights(world, states)
    features = [ helpersContinuous.getGaussianWeightFeatures(world, state) for state in states ]
    allFeatures.append(features[:-1])
    #print(gaussianWeights)
    
    for i in range(numobs-1):
        states, actions = helpersContinuous.doRollout(world, policyMatrix, 30)
        sumGaussianWeights += helpersContinuous.calculateGaussianWeights(world, states)
        features = [ helpersContinuous.getGaussianWeightFeatures(world, state) for state in states ]
        
        allStates.append(states[:-1])
        allActions.append(actions)
        allFeatures.append(features[:-1])

    helpersContinuous.exportObservations(allStates, allActions, allFeatures)

    ####### Reading obervations
    
    with open(obsfile, 'r') as infile:
        obsjson = json.load(infile)
    
    obsstates = obsjson["States"]
    obsactions = obsjson["Actions"]
    obsfeatures = obsjson["Features"]


    ####### Defining Korali Problem

    k = korali.Engine()
    e = korali.Experiment()
    
    ### Fixing the environment

    env = lambda s : worldEnv(s, N)

    ### Defining the Cartpole problem's configuration

    e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
    e["Problem"]["Possible Actions"] = [ [ 0.0 ], [ 1.0 ], [ 2.0 ], [ 3.0 ] ]
    e["Problem"]["Environment Function"] = env
    e["Problem"]["Training Reward Threshold"] = 600
    e["Problem"]["Policy Testing Episodes"] = 1
    e["Problem"]["Actions Between Policy Updates"] = 1

    e["Problem"]["Observations"]["States"] = obsstates
    e["Problem"]["Observations"]["Actions"] = obsactions
    e["Problem"]["Observations"]["Features"] = obsfeatures

    e["Variables"][0]["Name"] = "Position X"
    e["Variables"][0]["Type"] = "State"

    e["Variables"][1]["Name"] = "Position Y"
    e["Variables"][1]["Type"] = "State"

    e["Variables"][2]["Name"] = "Force"
    e["Variables"][2]["Type"] = "Action"

    ### Defining Agent Configuration 

    e["Solver"]["Type"] = "Agent / Discrete / DVRACER"
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Experiences Between Policy Updates"] = 1
    e["Solver"]["Episodes Per Generation"] = 1

    ### Defining the configuration of replay memory

    e["Solver"]["Experience Replay"]["Start Size"] = 1024
    e["Solver"]["Experience Replay"]["Maximum Size"] = 16384

    ## Defining Neural Network Configuration for Policy and Critic into Critic Container

    e["Solver"]["Discount Factor"] = 0.9
    e["Solver"]["Learning Rate"] = 1e-4
    e["Solver"]["Mini Batch"]["Size"] = 32
    e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False
    e["Solver"]["State Rescaling"]["Enabled"] = False

    ### IRL related configuration

    e["Solver"]["Experiences Between Reward Updates"] = 1
    e["Solver"]["Rewardfunction Learning Rate"] = 1e-5
    e["Solver"]["Demonstration Batch Size"] = 10
    e["Solver"]["Background Batch Size"] = 20
    e["Solver"]["Use Fusion Distribution"] = True
    e["Solver"]["Experiences Between Partition Function Statistics"] = 1e5

    ### Configuring the neural network and its hidden layers

    e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

    ### Defining Termination Criteria

    e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6

    ### Setting file output configuration

    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 10000
    e["File Output"]["Path"] = '_korali_results_discrete_15'

    ### Running Experiment

    k.run(e)
