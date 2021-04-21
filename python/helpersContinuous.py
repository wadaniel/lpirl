import numpy as np


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

    print("DVI terminated (iterations {}, max abs diff {}".format(it, maxDiffNorm))
    return valueMatrix, policyMatrix

def doRollout(gridworld, policy, maxsteps):
    gridworld.setPosition(0,0)
    currentPosition = gridworld.getPosition()
    stateList = [currentPosition]
    step = 0
    while step < maxsteps:
        x,y = tuple(currentPosition)
        nx = int(np.round(x*(gridworld.discretization-1)))
        ny = int(np.round(y*(gridworld.discretization-1)))
        action = policy[ (nx,ny) ]
        gridworld.move(action)
        currentPosition = gridworld.getPosition()
        stateList.append(currentPosition)
        step += 1

    return stateList


def doRolloutNoNoise(gridworld, policy, maxsteps):
    noise = gridworld.noise
    gridworld.noise = 0.0
    rollout = doRollout(gridworld, policy, maxsteps)
    gridworld.noise = noise
    return rollout


def createStateVectorView(gridworld, stateList):
    N = gridworld.discretization
    gamma = gridworld.discount
    stateVectorView = np.zeros(N**2)
    for idx, state in enumerate(stateList):
        stateIdx = state[0]+state[1]*N
        stateVectorView[stateIdx] += gamma**idx

    return stateVectorView


def createStateMatrixView(gridworld, stateList):
    N = gridworld.discretization
    gamma = gridworld.discount
    stateVectorView = np.zeros((N,N))
    for idx, state in enumerate(stateList):
        x,y = tuple(state)
        stateVectorView[x,y] += gamma**idx

    return stateVectorView


def calculateGaussianWeights(gridworld, rollout):
    N = gridworld.discretization
    gamma = gridworld.discount
   
    Xmat = []
    for i in range(N):
        Xmat.append(np.arange(0, N))
    
    Xmat = np.array(Xmat)/(N-1)
    Ymat = np.transpose(Xmat)
 
    sigma = 1.0/N
    scale = 1.0/(2.0*np.pi*sigma*sigma)
    weights = np.zeros((N,N))
    for idx, state in enumerate(rollout):
        weights += (gamma**idx)*scale*np.exp(-((state[0]-Xmat)/sigma)**2-((state[1]-Ymat)/sigma)**2)

    return weights


