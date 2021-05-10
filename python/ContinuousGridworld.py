import sys
import numpy as np
from scipy.stats import norm

import helpersContinuous
 
class ContinuousGridworld:
   
    '''
    @param self: Gridworld instance
    @param length: length of side of quadratic gridwold
    @param stepsize: step size in coordinate direction
    @param discretization: number of discretization vertices per dimension
    @param noise: uniform noise in coordinate directions
    @param discount: discount factor
    @param rewards: edges of subgrid with reward 1 (0 everywhere else)
    '''
    def __init__(self, length, stepsize, discretization, noise, discount, rewards):
        
        self.position = np.zeros(2)
        self.stepsize = stepsize
        self.noise = noise
        self.discount = discount
        self.discretization = discretization
        self.gaussianWeights = None
        if length == 1.0:
            self.length = length
        else:
            print("[ContinuousGridworld] length must be 1.0, is {}".format(length))

        rewardShape = rewards.shape
        if rewardShape == (2,2):
            self.rewards = rewards
        else:
            print("[ContinuousGridworld] rewards are of shape {}, but should be (2,2)".format(rewardShape))
            sys.exit()
  
  
    def getPosition(self):
        return self.position.copy()
       
    def setPosition(self, x, y):
        self.position = np.array([x, y])
 
    def setRandomPosition(self):
        x = np.random.uniform(0.0, self.length)
        y = np.random.uniform(0.0, self.length)
        self.position = np.array([x, y])

    def setGaussianWeights(self, weights):
        if weights.shape == (self.discretization, self.discretization):
            self.gaussianWeights = weights
        else:
            print("[ContinuousGridworld] weights are of shape {}, but should be {}".format(weights.shape, (self.discretization, self.discretization)))
            sys.exit()
 
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
            print("[ContinuousGridworld] action is {}, but should be 0, 1, 2 or 3".format(action))
        
        if action == 0:
            self.moveLeft()
        elif action == 1:
            self.moveRight()
        elif action == 2:
            self.moveUp()
        else:
            self.moveDown()

        if self.noise > 0:
            ux = np.random.uniform(-self.noise, +self.noise, 1)
            uy = np.random.uniform(-self.noise, +self.noise, 1)
            self.position[0] += ux
            self.position[1] += uy

        self.position = np.clip(self.position, 0, self.length)

    def moveLeft(self):
        self.position[0] -= self.stepsize
 
    def moveRight(self):
        self.position[0] += self.stepsize
    
    def moveUp(self):
        self.position[1] += self.stepsize
    
    def moveDown(self):
        self.position[1] -= self.stepsize
 
    def getReward(self):
        return self.getRewardAtPosition(self.position)
  
    def getRewardAtPosition(self, position):
        if self.gaussianWeights is None:
            if position[0] >= self.rewards[0, 0] and position[0] <= self.rewards[1, 0] and position[1] >= self.rewards[0, 1] and position[1] <= self.rewards[1, 1]:
                    return 1.0
            else:
                return 0.0

        else:
           
            Xmat = []
            for i in range(self.discretization):
                Xmat.append(np.arange(0, self.discretization))
 
            Xmat = np.array(Xmat)/(self.discretization-1)
            Ymat = np.transpose(Xmat)
 
            sigma = 1.0/self.discretization
            scale = 1.0/(2.0*np.pi*sigma*sigma)
            gweights = scale*np.exp(-((position[0]-Xmat)/sigma)**2-((position[1]-Ymat)/sigma)**2)


            return np.multiply(gweights, self.gaussianWeights).sum()



    def getProbabilityMatrixAtPosition(self, position):
        pMat = np.zeros((self.discretization, self.discretization))
        cleft = np.round((position[0]-self.noise)*self.discretization)
        cright = np.round((position[0]+self.noise)*self.discretization)
        cup = np.round((position[1]+self.noise)*self.discretization)
        cdown = np.round((position[1]-self.noise)*self.discretization)

        ncells = (1+cup-cdown)*(1+cright-cleft)
        for dx in range(int(1+cright-cleft)):
            x = int(np.clip(cleft+dx, 0, self.discretization-1))
            for dy in range(int(1+cup-cdown)):
                y = int(np.clip(cdown+dy, 0, self.discretization-1))
                
                pMat[x,y] += 1./ncells

        return pMat


if __name__ == "__main__":

    rewards = np.array([[0.8, 0.8], [1.0, 1.0]])
    
    print("Creating ContinuousGridworld ..")
    world = ContinuousGridworld(length=1.0, stepsize=0.2, discretization=5, noise=0.1, discount=0.9, rewards=rewards)
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
    
    pMat = world.getProbabilityMatrixAtPosition(world.position)
    print("Probability Matrix at Current Position:")
    print(pMat)


    print("Perform Value Iteration..")
    valueMatrix, policyMatrix = helpersContinuous.doDiscretizedValueIteration(world, 1e-3, 1e3, noisy=True)

    print("Policy Matrix:")
    print(policyMatrix)
    print("Value Matrix:")
    print(valueMatrix)

    sys.exit()
    print("Do Rollout..")
    rollout = helpersContinuous.doRollout(world, policyMatrix, 2*N)
    print("State List:")
    print(rollout)

    stateVectorView = helpersContinuous.createStateVectorView(world, rollout)
    print("State Vector View:")
    print(stateVectorView)
 
    stateMatrixView = helpersContinuous.createStateMatrixView(world, rollout)
    print("State Matrix View:")
    print(stateMatrixView)

    print("Exit..")
