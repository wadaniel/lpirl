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
    parser.add_argument('--noise', type=float, default=0.3, help='action noise')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')

    args = parser.parse_args()

    N = args.length
    p = args.noise
    gamma = args.discount
    maxiterations = args.iteration

    ## Initialization

    np.random.seed(args.seed)

    # create rewards

    rewards, terminal = helpers.createSinkReward(N, 1)
    
    # find optimal policy

    world = Gridworld(length=N, noise=p, discount=gamma, rewards=rewards, terminal=terminal)
    valueMatrix, policyMatrix = helpers.doValueIteration(world, 1e-3, 1e4)
    rollout = helpers.doRolloutNoNoise(world, policyMatrix, 2*N)
    stateVectorView = helpers.createStateVectoView(world, rollout)
 
    print(valueMatrix)
    print(policyMatrix)
    print(stateVectorView)


    ## Reconstruct rewards

    
    # initial reward weights

    c = np.random.uniform(0.,1.,N**2)
    cworld = Gridworld(length=N, noise=p, discount=gamma, rewards=c, terminal=terminal)
    
    cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, 1e-3, 1e3)
    crollout = helpers.doRollout(cworld, cpolicyMatrix, 2*N)
    cstateVectorView = helpers.createStateVectoView(cworld, crollout)

    #print(cvalueMatrix)
    #print(cpolicyMatrix)
    #print(cstateVectorView)


    ## Start IRL iteration ioncluding LP

    c = stateVectorView - cstateVectorView
    A = np.array([-c])
    b = np.zeros(1)
    alpha_bounds = (0.0, 1.0)
    bounds = [ alpha_bounds for i in range(N**2) ] 

    for it in range(maxiterations):

        #print(c)
        
        res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds)
        print(res)
        
        c = res.x
        cworld = Gridworld(length=N, noise=p, discount=gamma, rewards=c, terminal=terminal)
        cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, 1e-3, 1e3)
        crollout = helpers.doRollout(cworld, cpolicyMatrix, 2*N)
        cstateVectorView = helpers.createStateVectoView(cworld, crollout)
 
        #print(cvalueMatrix)
        #print(cpolicyMatrix)
        #print(cstateVectorView)

        cdiff = stateVectorView - cstateVectorView
        print(cdiff)
        c += cdiff
        A = np.append(A, [-cdiff], axis=0)
        #print(A)
        b = np.zeros(it+2)


    print(-A)
