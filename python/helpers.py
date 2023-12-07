import numpy as np

def createSinkReward(gridLength, maxval):
    rewards = np.zeros((gridLength, gridLength))
    rewards[gridLength-1, gridLength-1] = maxval
    terminal = np.zeros((gridLength, gridLength))
    terminal[gridLength-1, gridLength-1] = 1
    return rewards, terminal

def createDoubleSinkReward(gridLength, maxval):
    rewards = np.zeros((gridLength, gridLength))
    rewards[gridLength-1, gridLength-1] = maxval
    rewards[gridLength-1, 0] = maxval
    terminal = np.zeros((gridLength, gridLength))
    terminal[gridLength-1, gridLength-1] = 1
    terminal[gridLength-1, 0] = 1
    return rewards, terminal

def createTripleSinkReward(gridLength, maxval):
    rewards = np.zeros((gridLength, gridLength))
    rewards[gridLength-1, gridLength-1] = maxval
    rewards[gridLength-1, 0] = maxval
    rewards[0, gridLength-1] = maxval
    terminal = np.zeros((gridLength, gridLength))
    terminal[gridLength-1, gridLength-1] = 1
    terminal[gridLength-1, 0] = 1
    terminal[0, gridLength-1] = 1
    return rewards, terminal

def doValueIteration(gridworld, eps, maxsteps, noisy=False):
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
                    leftPosition = tuple(np.clip(np.array([x,y-1]), 0, N-1))
                    rightPosition = tuple(np.clip(np.array([x,y+1]), 0, N-1))
                    upPosition = tuple(np.clip(np.array([x-1,y]), 0, N-1))
                    downPosition = tuple(np.clip(np.array([x+1,y]), 0, N-1))
                
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

        if noisy:
            print("Iteration {}: max abs diff {}".format(it, maxDiffNorm))
        
        valueMatrix = valueMatrixCopy.copy()
        it += 1

    print(f"Terminate with {it}/{maxsteps} iterations and norm {maxDiffNorm}/{eps}")
    return valueMatrix, policyMatrix


def doRollout(gridworld, policy, maxsteps):
    #gridworld.setPosition(0,0)
    gridworld.setRandomPosition()
    isTerminal = gridworld.isTerminal()
    currentPosition = gridworld.getPosition()
    stateList = [currentPosition]
    step = 0
    while isTerminal == 0 and step < maxsteps:
        x,y = tuple(currentPosition)
        action = policy[ (x,y) ]
        gridworld.move(action)
        currentPosition = gridworld.getPosition()
        isTerminal = gridworld.isTerminal()
        stateList.append(currentPosition)
        step += 1

    return stateList

def doRolloutNoNoise(gridworld, policy, maxsteps):
    noise = gridworld.noise
    gridworld.noise = 0.0
    stateList = doRollout(gridworld, policy, maxsteps)
    gridworld.noise = noise
    return stateList


def createStateVectorView(gridworld, stateList):
    N = gridworld.length
    gamma = gridworld.discount
    stateVectorView = np.zeros(N**2)
    for idx, state in enumerate(stateList):
        stateIdx = state[0]+state[1]*N
        stateVectorView[stateIdx] += gamma**idx

    return stateVectorView

def createStateMatrixView(gridworld, stateList):
    N = gridworld.length
    gamma = gridworld.discount
    stateVectorView = np.zeros((N,N))
    for idx, state in enumerate(stateList):
        x,y = tuple(state)
        stateVectorView[x,y] += gamma**idx

    return stateVectorView

def printPolicyMatrix(policy):
    N1, N2 = policy.shape
    charPolicy = np.full((N1,N2),'.')

    for i in range(N1):
        for j in range(N2):
            if policy[i,j] == 0:
                charPolicy[i,j] = '<'

            elif policy[i,j] == 1:
                charPolicy[i,j] = '>'

            elif policy[i,j] == 2:
                charPolicy[i,j] = '^'

            elif policy[i,j] == 3:
                charPolicy[i,j] = 'v'

    print(charPolicy)
