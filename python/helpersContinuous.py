import numpy as np
from scipy.stats import norm


def evaluateGaussianGrid(length, weightsPerDim, position):
    dx = length/(weightsPerDim-1.)
    sigma = dx
    evaluation = np.zeros((weightsPerDim, weightsPerDim))
    for i in range(weightsPerDim):
        for j in range(weightsPerDim):
            x = i*dx
            y = j*dx
            evaluation[i,j] = norm.pdf(position[0], x, sigma) * norm.pdf(position[1], y, sigma)

    return evaluation


def doDiscretizedValueIteration(gridworld, eps, maxsteps, noisy=False):
    N = gridworld.discretization
    stepSize = gridworld.stepsize
    p = gridworld.noise
    gamma = gridworld.discount
    valueMatrix = np.random.normal(0, 1, (N, N))
    valueMatrixCopy = valueMatrix.copy()
    
    it = 0
    maxDiffNorm = np.Inf
    while it < maxsteps and maxDiffNorm > eps:
        
        policyMatrix = np.zeros((N,N))
        for nx in range(N):
            for ny in range(N):

                x = nx/(N-1)
                y = ny/(N-1)

                # Positions

                leftPosition = tuple(np.clip(np.array([x-stepSize,y]), 0, 1.0))
                rightPosition = tuple(np.clip(np.array([x+stepSize,y]), 0, 1.0))
                upPosition = tuple(np.clip(np.array([x,y+stepSize]), 0, 1.0))
                downPosition = tuple(np.clip(np.array([x,y-stepSize]), 0, 1.0))
 
                # Probability Matrices

                leftP = gridworld.getProbabilityMatrixAtPosition(leftPosition)
                rightP = gridworld.getProbabilityMatrixAtPosition(rightPosition)
                upP = gridworld.getProbabilityMatrixAtPosition(upPosition)
                downP = gridworld.getProbabilityMatrixAtPosition(downPosition)

                # Current Reward

                reward = gridworld.getRewardAtPosition([x,y])
                
                # Value Matrix

                leftValue = np.multiply(valueMatrix, leftP).sum()
                rightValue = np.multiply(valueMatrix, rightP).sum()
                upValue = np.multiply(valueMatrix, upP).sum()
                downValue = np.multiply(valueMatrix, downP).sum()

                values = np.array([leftValue, rightValue, upValue, downValue])

                # Q(s,a) 

                q = reward + gamma * values
                
                # Evaluate best action

                bestAction = np.argmax(q)

                # Set new value
        
                policyMatrix[(nx,ny)] = bestAction
                valueMatrixCopy[(nx,ny)] = q[bestAction]

        diffMatrix = abs(valueMatrixCopy - valueMatrix)
        maxDiffNorm = diffMatrix.max()

        if noisy:
            print("Iteration {}: max abs diff {}".format(it, maxDiffNorm))
        
        valueMatrix = valueMatrixCopy.copy()
        it += 1

    return valueMatrix, policyMatrix


