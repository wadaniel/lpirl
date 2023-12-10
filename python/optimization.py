from Gridworld import *
import helpers

from scipy.optimize import linprog
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1, help='number irl iterations')
    parser.add_argument('--length', type=int, default=8, help='length of Gridworld')
    parser.add_argument('--discount', type=float, default=0.95, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='accuracy of value iteration')
    parser.add_argument('--noise', type=float, default=0.2, help='probability of random action')
    parser.add_argument('--numobs', type=int, default=1, help='number observed expert trajectories')
    parser.add_argument('--pFactor', type=float, default=2.0, help='penalization constraint')
    parser.add_argument('--lam', type=float, default=1.05, help='penalization reward weights')
    parser.add_argument('--sinks', type=int, default=1, help='number of reward states')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')

    args = parser.parse_args()

    N = args.length
    p = args.noise
    gamma = args.discount
    epsilon = args.epsilon
    maxiterations = args.iteration
    numobs = args.numobs
    pFactor = args.pFactor
    lam = args.lam
    sinks = args.sinks

    ## Initialization

    np.random.seed(args.seed)

    # create rewards

    if sinks == 1:
        rewards, terminal = helpers.createSinkReward(N, 1)
    elif sinks == 2:
        rewards, terminal = helpers.createDoubleSinkReward(N, 1)
    elif sinks == 3:
        rewards, terminal = helpers.createTripleSinkReward(N, 1)
    else:
        print(f"[optimization] number of sinks {sinks} not valid")
   
    # find optimal policy

    world = Gridworld(length=N, noise=p, discount=gamma, rewards=rewards, terminal=terminal)
    valueMatrix, policyMatrix = helpers.doValueIteration(world, epsilon, 1e4)

    helpers.printPolicyMatrix(policyMatrix)

    #rollout = helpers.doRolloutNoNoise(world, policyMatrix, 2*N-1)
    rollout = helpers.doRollout(world, policyMatrix, N**2)
    stateVectorView = helpers.createStateVectorView(world, rollout)
    stateMatrixView = helpers.createStateMatrixView(world, rollout)
 
    rollouts = [rollout]
    for i in range(1, numobs):
        rollout = helpers.doRollout(world, policyMatrix, N**2)
        stateVectorView += helpers.createStateVectorView(world, rollout)
        rollouts.append(rollout)

    #print(rewards)
    #print(terminal)
    #print(valueMatrix)
    #print(policyMatrix)
    #print(stateVectorView)
    #print(stateMatrixView)

    ## Reconstruct rewards
    
    # initial reward weights

    crewards = np.random.uniform(0.,1.,(N,N))
    cworld = Gridworld(length=N, noise=p, discount=gamma, rewards=crewards, terminal=terminal)
    
    cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, epsilon, 1e3)
    # crollout = helpers.doRolloutNoNoise(cworld, cpolicyMatrix, 2*N-1)
    crollout = helpers.doRollout(cworld, cpolicyMatrix, N**2)
    cstateVectorView = helpers.createStateVectorView(cworld, crollout)

    #print(cvalueMatrix)
    #print(cpolicyMatrix)
    #print(cstateVectorView)


    ## Start IRL iteration including LP

    cdiff = stateVectorView - numobs*cstateVectorView
    cdiff[cdiff < 0] = pFactor*cdiff[ cdiff < 0] 
    c = cdiff
    A = np.array([-cdiff])
    b = np.zeros(1)
    alpha_bounds = (0.0, 1.0)
    bounds = [ alpha_bounds for i in range(N**2) ] 

    crewards = [crewards]
    cvalues = [cvalueMatrix]
    cpolicies = [cpolicyMatrix]
    errors = [(cpolicyMatrix != policyMatrix).sum()]

    for it in range(0,maxiterations):

        print("[IRL] Iteration {}".format(it))
        #print(c)
        
        tmp = c-lam
        res = linprog(-tmp, A_ub=None, b_ub=None, bounds=bounds)
        # res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds)
        #print(res)
        
        crewardMatrix = np.transpose(np.reshape(res.x, (N,N)))
        cworld = Gridworld(length=N, noise=p, discount=gamma, rewards=crewardMatrix, terminal=terminal)
        cvalueMatrix, cpolicyMatrix = helpers.doValueIteration(cworld, epsilon, 1e3)
        crollout = helpers.doRollout(cworld, cpolicyMatrix, N**2)
        cstateVectorView = helpers.createStateVectorView(cworld, crollout)

        for rollout in rollouts:
            crollout = helpers.doRollout(cworld, cpolicyMatrix, N**2, initPos=(rollout[0][0], rollout[0][1]))
            cstateVectorView += helpers.createStateVectorView(cworld, crollout)
 
        #cdiff = stateVectorView - numobs*cstateVectorView
        cdiff = stateVectorView - cstateVectorView
        cdiff[cdiff < 0] = cdiff[cdiff < 0]*2
        c += cdiff

        crewards.append(crewardMatrix)
        cvalues.append(cvalueMatrix)
        cpolicies.append(cpolicyMatrix)
        errors.append((cpolicyMatrix != policyMatrix).sum())
        print(errors[-1])

        # Constraints
        # A = np.append(A, [-cdiff], axis=0)
        # b = np.zeros(it+2)

    rollouts = np.array(rollouts)
    cvalues = np.array(cvalues)
    cpolicies = np.array(cpolicies)
    crewards = np.array(crewards)
    errors = np.array(errors)
 
    print("Expert Return Matrix")
    print(rewards)

    print("Expert Value Matrix")
    print(valueMatrix)

    print("Expert Policy Matrix")
    print(policyMatrix)
    helpers.printPolicyMatrix(policyMatrix)

    print("Expert State Vector View")
    print(stateVectorView)

    print("IRL Return Matrix")
    print(crewardMatrix)

    print("IRL Value Matrix")
    print(cvalueMatrix)
    
    print("IRL Policy Matrix")
    print(cpolicyMatrix)
    helpers.printPolicyMatrix(cpolicyMatrix)
    
    print("IRL C")
    print(c)

    np.savez('./output/optimization.npz', rewards=rewards, valMat=valueMatrix, 
             polMat=policyMatrix, cvalueMatrix=cvalueMatrix, 
             cpolicyMatrix=cpolicyMatrix, crewardMat=crewardMatrix, 
             cvalues=cvalues, cpolicies=cpolicies, 
             crewards=crewards, errors=errors, terminal=terminal, rollouts=rollouts)
