import numpy as np

def createSinkReward(gridLength, maxval):
    rewards = np.zeros(gridLength**2)
    rewards[-1] = maxval
    terminal = np.zeros(gridLength**2)
    terminal[-1] = 1
    return rewards, terminal


def doValueIteration(gridworld, eps, maxsteps):
    N = gridworld.length
    p = gridworld.noise
    gamma = gridworld.discount
    valueMatrix = np.random.normal(0, 1, (N, N))
    valueMatrixCopy = valueMatrix.copy()
    
    leftP = np.array([1.-3./4.*p, p/4., p/4., p/4.])
    rightP = np.array([p/4., 1.-3./4.*p, p/4., p/4.])
    upP = np.array([p/4., p/4., 1.-3./4.*p, p/4.])
    downP = np.array([p/4., p/4., p/4., 1.-3./4.*p])


    it = 0
    maxDiffNorm = np.Inf
    while it < maxsteps and maxDiffNorm > eps:
        
        policyMatrix = np.zeros((N,N))
        for x in range(N):
            for y in range(N):

                # Check status of state

                isTerminal = gridworld.getTerminationAtPosition([x,y])

                if isTerminal:
                    policyMatrix[(x,y)] = -1
                    valueMatrixCopy[(x,y)] = gridworld.getRewardAtPosition([x,y])

                else:
                    
                    # Positions

                    leftPosition = tuple(np.clip(np.array([x-1,y]), 0, N-1))
                    rightPosition = tuple(np.clip(np.array([x+1,y]), 0, N-1))
                    upPosition = tuple(np.clip(np.array([x,y+1]), 0, N-1))
                    downPosition = tuple(np.clip(np.array([x,y-1]), 0, N-1))
                
                    # Current Reward

                    reward = gridworld.getRewardAtPosition([x,y])
                    
                    # Values

                    leftValue = valueMatrix[leftPosition]
                    rightValue = valueMatrix[rightPosition]
                    upValue = valueMatrix[upPosition]
                    downValue = valueMatrix[downPosition]

                    values = np.array([leftValue, rightValue, upValue, downValue])

                    # Q(s,a) 

                    q = np.zeros(4)
                    
                    q[0] = reward + gamma * np.dot(leftP, values)
                    q[1] = reward + gamma * np.dot(rightP, values)
                    q[2] = reward + gamma * np.dot(upP, values)
                    q[3] = reward + gamma * np.dot(downP, values)

                    # Evaluate best action

                    bestAction = np.argmax(q)

                    # Set new value
            
                    policyMatrix[(x,y)] = bestAction
                    valueMatrixCopy[(x,y)] = q[bestAction]

        diffMatrix = abs(valueMatrixCopy - valueMatrix)
        maxDiffNorm = diffMatrix.max()

        print("Iteration {}: max abs diff {}".format(it, maxDiffNorm))
        valueMatrix = valueMatrixCopy.copy()
        it += 1

    return valueMatrix, policyMatrix


def doRollout(gridworld, policy):
    gridworld.setPosition(0,0)
    isTerminal = gridworld.isTerminal()
    currentPosition = gridworld.getPosition()
    stateList = [currentPosition]
    while isTerminal == 0:
        action = policy[ tuple(currentPosition) ]
        gridworld.move(action)
        currentPosition = gridworld.getPosition()
        isTerminal = gridworld.isTerminal()
        stateList.append(currentPosition)

    return stateList

def createStateVectoView(gridworld, stateList):
    N = gridworld.length
    gamma = gridworld.discount
    stateVectorView = np.zeros(N**2)
    for idx, state in enumerate(stateList):
        stateIdx = state[1]*N+state[0]
        stateVectorView[stateIdx] += gamma**idx

    return stateVectorView