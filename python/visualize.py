import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


fdir = 'figures'
fileName = './output/optimization.npz'

arrows = {-1:(0,0), 1:(1,0), 0:(-1,0),2:(0,1),3:(0,-1)}

def plotPolicy(polMat, terminal, fname, truePolMat=None):
    scale = 0.25

    fig, ax = plt.subplots(figsize=(6, 6))
    for r, row in enumerate(polMat):
        for c, cell in enumerate(row):
            if terminal[r,c] == 0:
                plt.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.2, color='cornflowerblue')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'{fdir}/{fname}.pdf')
    plt.close()


def plotPolicyAx(ax, polMat, terminal, truePolMat):
    scale = 0.25

    for r, row in enumerate(polMat):
        for c, cell in enumerate(row):
            if terminal[r,c] == 0:
                if truePolMat[r,c] == polMat[r,c]:
                    ax.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.2, color='cornflowerblue')
                else:
                    ax.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.2, color='lightcoral')

    ax.set_xticks([])
    ax.set_yticks([])



def convergencePlots(crewards, cvalues, cpolicies, errors, terminal, polMat):
    col=sns.light_palette("seagreen", as_cmap=True)

    N1, _ = polMat.shape
    N = len(errors)

    for idx in range(N):
        print(f"{idx}/{N}")
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        sns.heatmap(crewards[idx], ax=axes[0], linewidths=1.0, square=True, cmap=col, cbar=False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        sns.heatmap(cvalues[idx], ax=axes[1], linewidths=1.0, square=True, cmap=col, cbar=False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plotPolicyAx(axes[2], cpolicies[idx], terminal, polMat)
        sns.lineplot(errors[:idx+1], ax=axes[3], color='lightcoral')
        axes[3].set_ylim(0,N1)
        axes[3].set_xlim(0,N)
        axes[3].set_xticks(range(0,N,10))
        axes[3].set_yticks([0,N1**2//2,N1**2])
        plt.savefig(f'{fdir}/cplots_{idx:03d}.png')
        plt.close()

def rolloutPlot(rollouts):
    plt.figure()
    for rollout in rollouts:
        plt.plot(rollout[:,1], rollout[:,0])

    plt.show()
    plt.close()

data = np.load(fileName, allow_pickle=True)
rewards = data["rewards"]
terminal = data["terminal"]
valMat = data["valMat"]
polMat = data["polMat"]
crewMat = data["crewardMat"]
cvalMat = data["valMat"]
cpolMat = data["polMat"]
cvalues = data["cvalues"]
crewards = data["crewards"]
cpolicies = data["cpolicies"]
errors = data["errors"]
rollouts = data["rollouts"]

col=sns.light_palette("seagreen", as_cmap=True)

#rolloutPlot(rollouts)
#exit()

convergencePlots(crewards, cvalues, cpolicies, errors, terminal, polMat)

sns.heatmap(rewards, linewidths=1.0, square=True, cmap=col, cbar=False)
plt.xticks([])
plt.yticks([])
plt.savefig(f'{fdir}/rewards.pdf')
plt.close()

sns.heatmap(valMat, linewidths=1.0, square=True, cmap=col)
plt.xticks([])
plt.yticks([])
plt.savefig(f'{fdir}/valMat.pdf')
plt.close()

plotPolicy(polMat, terminal, 'polMat')

"""
sns.heatmap(crewMat, linewidths=1.0, square=True, cmap=col)
plt.savefig(f'{fdir}/crefMat.pdf')
plt.close()

sns.heatmap(cvalMat, linewidths=1.0, square=True, cmap=col)
plt.savefig(f'{fdir}/cvalMat.pdf')
plt.close()

sns.heatmap(cvalues[0], linewidths=1.0, square=True, cmap=col)
plt.show()
plt.close()

sns.heatmap(crewards[0], linewidths=1.0, square=True, cmap=col)
plt.show()
plt.close()
"""
