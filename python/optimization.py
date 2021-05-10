from Gridworld import *
import helpers

from scipy.optimize import linprog
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1, help='number irl iterations')
    parser.add_argument('--length', type=int, default=5, help='length of Gridworld')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='accuracy of value iteration')
    parser.add_argument('--noise', type=float, default=0.3, help='action noise')
    parser.add_argument('--numobs', type=int, default=1, help='number observed expert trajectories')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')

    args = parser.parse_args()

    N = args.length
    p = args.noise
    gamma = args.discount
    epsilon = args.epsilon
    maxiterations = args.iteration
    numobs = args.numobs

    ## Initialization

    np.random.seed(args.seed)

    # create rewards

    rewards, terminal = helpers.createSinkReward(N, 1)
    
    # find optimal policy

    world = Gridworld(length=N, noise=p, discount=gamma, rewards=rewards, terminal=terminal)
    valueMatrix, policyMatrix = helpers.doValueIteration(world, epsilon, 1e4)
 
    #rollout = helpers.doRolloutNoNoise(world, policyMatrix, 2*N-1)
    rollout = helpers.doRollout(world, policyMatrix, 2*N-1)
    stateVectorView = helpers.createStateVectorView(world, rollout)
    stateMatrixView = helpers.createStateMatrixView(world, rollout)
 
    for i in range(numobs-1):
        rollout = helpers.doRollout(world, policyMatrix, 2*N-1)
        stateVectorView += helpers.createStateVectorView(world, rollout)
 
    print(valueMatrix)
    print(policyMatrix)
    print(stateVectorView)
    print(stateMatrixView)

    ## Reconstruct rewards

    
    # initial reward weights

    crewards = np.random.uniform(0.,1.,(N,N))
    cworld = Gridworld(length=N, noise=p, discount=gamma, rewards=crewards, terminal=terminal)
    
    cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, epsilon, 1e3)
    # crollout = helpers.doRolloutNoNoise(cworld, cpolicyMatrix, 2*N-1)
    crollout = helpers.doRollout(cworld, cpolicyMatrix, 2*N)
    cstateVectorView = helpers.createStateVectorView(cworld, crollout)

    #print(cvalueMatrix)
    #print(cpolicyMatrix)
    #print(cstateVectorView)


    ## Start IRL iteration including LP

    cdiff = stateVectorView - numobs*cstateVectorView
    cdiff[cdiff < 0] = 2.0*cdiff[ cdiff < 0] 
    c = cdiff
    A = np.array([-cdiff])
    b = np.zeros(1)
    alpha_bounds = (0.0, 1.0)
    bounds = [ alpha_bounds for i in range(N**2) ] 

    for it in range(maxiterations):

        print("[IRL] Iteration {}".format(it))
        #print(c)
        
        # res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds)
        res = linprog(-c, A_ub=None, b_ub=None, bounds=bounds)
        #print(res)
        
        crewards = np.transpose(np.reshape(res.x, (N,N)))
        #crewards = np.reshape(res.x, (N,N))
        print(crewards)
        cworld = Gridworld(length=N, noise=p, discount=gamma, rewards=crewards, terminal=terminal)
        cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, epsilon, 1e3)
        crollout = helpers.doRollout(cworld, cpolicyMatrix, 2*N-1)
        cstateVectorView = helpers.createStateVectorView(cworld, crollout)
 
        #print(cvalueMatrix)
        #print(cpolicyMatrix)
        #print(cstateVectorView)

        cdiff = stateVectorView - numobs*cstateVectorView
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

    print("Expert State Vector View")
    print(stateVectorView)

    print("IRL Value Matrix")
    print(cvalueMatrix)
    
    print("IRL Policy Matrix")
    print(cpolicyMatrix)
    
    print("IRL C")
    print(c)
