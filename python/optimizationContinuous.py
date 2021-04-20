from Gridworld import *
import helpers

from scipy.optimize import linprog
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1, help='number irl iterations')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor')
    parser.add_argument('--noise', type=float, default=0.3, help='action noise')
    parser.add_argument('--discretization', type=float, default=5, help='action noise')
    parser.add_argument('--numobs', type=int, default=1, help='number observed expert trajectories')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')

    args = parser.parse_args()

    N = args.discretization
    p = args.noise
    gamma = args.discount
    maxiterations = args.iteration
    numobs = args.numobs

    ## Initialization

    np.random.seed(args.seed)

    # create reward quadrant

    rewards = np.array([[0.8, 0.8], [1.0, 1.0]])
    
    # find optimal policy

    world = ContinuousGridworld(length=1.0, stepsize=0.2, discretization=N, noise=noise, discount=discount, rewards=rewards)
    valueMatrix, policyMatrix = helpers.doDiscretizedValueIteration(world, 1e-3, 1e4)
 
    sys.exit()
    rollout = helpers.doRolloutNoNoise(world, policyMatrix, 2*N-1)
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
    
    cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, 1e-3, 1e3)
    # crollout = helpers.doRolloutNoNoise(cworld, cpolicyMatrix, 2*N-1)
    crollout = helpers.doRollout(cworld, cpolicyMatrix, 2*N)
    cstateVectorView = helpers.createStateVectorView(cworld, crollout)

    #print(cvalueMatrix)
    #print(cpolicyMatrix)
    #print(cstateVectorView)


    ## Start IRL iteration including LP

    cdiff = stateVectorView - cstateVectorView
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
        cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, 1e-3, 1e3)
        crollout = helpers.doRollout(cworld, cpolicyMatrix, 2*N-1)
        cstateVectorView = helpers.createStateVectorView(cworld, crollout)
 
        #print(cvalueMatrix)
        #print(cpolicyMatrix)
        #print(cstateVectorView)

        cdiff = stateVectorView - cstateVectorView
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
