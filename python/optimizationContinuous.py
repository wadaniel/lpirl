from ContinuousGridworld import *
import helpersContinuous

from scipy.optimize import linprog
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1, help='number irl iterations')
    parser.add_argument('--discount', type=float, default=0.90, help='discount factor')
    parser.add_argument('--noise', type=float, default=0.3, help='action noise')
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

    ## Initialization

    np.random.seed(args.seed)

    # create reward quadrant

    rewards = np.array([[0.8, 0.8], [1.0, 1.0]])
    
    # find optimal policy

    world = ContinuousGridworld(length=1.0, stepsize=0.2, discretization=N, noise=p, discount=gamma, rewards=rewards)
    valueMatrix, policyMatrix = helpersContinuous.doDiscretizedValueIteration(world, epsilon, 1e4, noisy=noisy)
 
    print(valueMatrix)
    print(policyMatrix)

    rollout = helpersContinuous.doRollout(world, policyMatrix, 30)
    gaussianWeights = helpersContinuous.calculateGaussianWeights(world, rollout)

    print(gaussianWeights)

    #for i in range(numobs-1):
    #    rollout = helpersContinuous.doRollout(world, policyMatrix, 30)
    #    stateVectorView += helpersContinuous.createStateVectorView(world, rollout)

    ## Reconstruct rewards
    
    # initial reward weights

    cworld = ContinuousGridworld(length=1.0, stepsize=0.2, discretization=N, noise=p, discount=gamma, rewards=rewards)
    
    cweights = np.random.uniform(0.,1.,(N,N))
    cworld.setGaussianWeights(cweights)

    cvalueMatrix, cpolicyMatrix = helpersContinuous.doDiscretizedValueIteration(cworld, epsilon, 1e3, noisy=noisy)

    # crollout = helpersContinuous.doRolloutNoNoise(cworld, cpolicyMatrix, 2*N-1)
    crollout = helpersContinuous.doRollout(cworld, cpolicyMatrix, 30)
    
    cgaussianWeights = helpersContinuous.calculateGaussianWeights(cworld, crollout)

    print(cvalueMatrix)
    print(cpolicyMatrix)
    print(cgaussianWeights)

    ## Start IRL iteration including LP

    cdiff = gaussianWeights.flatten() - cgaussianWeights.flatten()
    c = cdiff
    A = np.array([-cdiff])
    b = np.zeros(1)
    alpha_bounds = (0.0, 1.0)
    bounds = [ alpha_bounds for i in range(N**2) ] 

    for it in range(maxiterations):

        print("[IRL] Iteration {}".format(it))
        
        res = linprog(-c, A_ub=None, b_ub=None, bounds=bounds)
        #print(res)
        
        #cweights = np.transpose(np.reshape(res.x, (N,N)))
        cweights = np.reshape(res.x, (N,N))
        print(cweights)
        cworld = ContinuousGridworld(length=1.0, stepsize=0.2, discretization=N, noise=p, discount=gamma, rewards=rewards)
        cworld.setGaussianWeights(cweights)

        cvalueMatrix, cpolicyMatrix = helpersContinuous.doDiscretizedValueIteration(cworld, epsilon, 1e3, noisy=noisy)
        crollout = helpersContinuous.doRollout(cworld, cpolicyMatrix, 30)
        cgaussianWeights = helpersContinuous.calculateGaussianWeights(cworld, crollout)
        #print(cvalueMatrix)
        #print(cpolicyMatrix)
        #print(cstateVectorView)

        cdiff = gaussianWeights.flatten() - cgaussianWeights.flatten()
        cdiff[cdiff < 0] = cdiff[cdiff < 0]*2
        c += cdiff

        # Constraints
        # A = np.append(A, [-cdiff], axis=0)
        # b = np.zeros(it+2)
 
    print("Expert Return Matrix")
    print(rewards)

    print("Expert Value Matrix")
    print(valueMatrix)

    print("Expert Policy Matrix")
    print(policyMatrix)

    print("Expert Gaussian Weights")
    print(gaussianWeights)

    print("IRL Value Matrix")
    print(cvalueMatrix)
    
    print("IRL Policy Matrix")
    print(cpolicyMatrix)
  
    print("Cweights")
    print(cweights)
    
    print("IRL C")
    print(c)
