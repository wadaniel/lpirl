import sys
import numpy as np

import helpers
"""
        [ (0,0)   ..    (0,N-1) ]
        [   ..    ..      ..    ]
        [ (N-1,0) ..      ..    ]
"""
 
class Gridworld:
   
    '''
    @param self: Gridworld instance
    @param length: length of side of quadratic gridwold
    @param noise: probability of random action
    @param discount: discount factor
    @param rewards: vector of rewards of states
    @param terminal: (0,1) vector indicating terminal states
    '''
    def __init__(self, length, noise, discount, rewards, terminal):
        
        self.position = np.zeros(2)
        self.noise = noise
        self.discount = discount

        if length > 0:
            self.length = length
        else:
            print("[Gridworld] length must be larger 0, is {}".format(length))

        if rewards.shape == (length,length):
            self.rewards = rewards
        else:
            print("[Gridworld] rewards are of shape {}, but should be {}".format(rewards.shape, (length,length)))
            sys.exit()
  
        if terminal.shape == (length, length):
            self.terminal = terminal
        else:
            print("[Gridworld] terminal are of shape {}, but should be {}".format(terminal.shape, (length, length)))
            sys.exit()
  
    def getPosition(self):
        return self.position.copy()
       
    def setPosition(self, x, y):
        self.position = np.array([x, y])

    def setRandomPosition(self):
        self.position = np.random.randint(0,self.length-1,2)
    '''
    @param self: Gridworld instance
    @param action: move direction: 
        0 left
        1 right
        2 up
        3 down
    '''
    def move(self, action):
        if action < 0 or action > 3:
            print("[Gridworld] action is {}, but should be 0, 1, 2 or 3".format(action))
        
        u = np.random.uniform(0.0, 1.0, 1)
        if u <= self.noise:
            # take a random step
            action = np.random.randint(0, 4, 1)

        if action == 0:
            self.moveLeft()
        elif action == 1:
            self.moveRight()
        elif action == 2:
            self.moveUp()
        elif action == 3:
            self.moveDown()

    def moveLeft(self):
        if self.position[1] > 0:
            self.position[1] -= 1
 
    def moveRight(self):
        if self.position[1] < self.length-1:
            self.position[1] += 1
    
    def moveUp(self):
        if self.position[0] > 0:
            self.position[0] -= 1
    
    def moveDown(self):
        if self.position[0] < self.length-1:
            self.position[0] += 1
 
    def getReward(self):
        reward = self.rewards[tuple(self.position)]
        return reward
 
    def getRewardAtPosition(self, position):
        reward = self.rewards[tuple(position)]
        return reward

    def isTerminal(self):
        isTerminal = self.terminal[tuple(self.position)]
        #return 0
        return isTerminal

    def getTerminationAtPosition(self, position):
        isTerminal = self.terminal[tuple(position)]
        #return 0
        return isTerminal



if __name__ == "__main__":

    N = 5
    p = 0.0
    rewards, terminal = helpers.createSinkReward(N, 10)
    
    print("Creating Gridworld ..")
    world = Gridworld(length=N, noise=p, discount=0.99, rewards=rewards, terminal=terminal)
    print("World created ..")
    print("Initial Position: {}".format(world.position))
    
    print("Move Left")
    world.move(0)
    print("Current Position: {}".format(world.position))
    print("Move Right")
    world.move(1)
    print("Current Position: {}".format(world.position))
    print("Move Up")
    world.move(2)
    print("Current Position: {}".format(world.position))
    print("Move Down")
    world.move(3)
    print("Current Position: {}".format(world.position))

    print("Perform Value Iteration..")
    valueMatrix, policyMatrix = helpers.doValueIteration(world, 1e-3, 1e3)

    print("Reward Matrix:")
    print(rewards)
    print("Value Matrix:")
    print(valueMatrix)
    print("Policy Matrix:")
    print(policyMatrix)
 

    print("Do Rollout..")
    rollout = helpers.doRollout(world, policyMatrix, 2*N-1)
    print("State List:")
    print(rollout)

    stateVectorView = helpers.createStateVectorView(world, rollout)
    print("State Vector View:")
    print(stateVectorView)
 
    stateMatrixView = helpers.createStateMatrixView(world, rollout)
    print("State Matrix View:")
    print(stateMatrixView)


    print("Exit..")
