import matplotlib.pyplot as plt
import numpy as np


def PlotRegret(optRewards, expRewards, tasks=1):
    meanExpectedRewards = np.mean(expRewards, axis=0)
    regret = np.cumsum(optRewards - meanExpectedRewards)

    regretColor = 'tab:red'
    optimalColor = 'tab:orange'
    expColor = 'tab:blue'

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Horizon')
    if tasks == 1:
        ax1.set_ylabel('Number of clicks', color=regretColor)
    else:
        ax1.set_ylabel('Regret (€)', color=regretColor)
    ax1.plot(regret, regretColor, label='Cumulative Regret')
    ax1.tick_params(axis='y', labelcolor=regretColor)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()
    if tasks == 1:
        ax2.set_ylabel('Number of clicks', color=expColor)
    else:
        ax2.set_ylabel('Reward (€)', color=expColor)
    ax2.plot(optRewards, optimalColor, label='Optimal Reward')
    ax2.plot(meanExpectedRewards, expColor, label='Expected Reward')
    ax2.tick_params(axis='y', labelcolor=expColor)
    ax2.legend(loc='lower right')

    fig.tight_layout()
    plt.show()


def PlotSWRegret(optRewards, expRewards, swExpRewards):
    meanExpectedRewards = np.mean(expRewards, axis=0)
    meanExpectedRewardsSW = np.mean(swExpRewards, axis=0)
    regret = np.cumsum(optRewards - meanExpectedRewards)
    regretSW = np.cumsum(optRewards - meanExpectedRewardsSW)

    regretColor = 'tab:red'
    optimalColor = 'tab:orange'
    expColor = 'tab:blue'

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].set_xlabel('Horizon')
    axs[0].set_ylabel('Number of clicks')
    axs[0].plot(regret, regretColor, label='Cumulative Regret')
    axs[0].plot(regretSW, regretColor, label='Cumulative Regret SW', linestyle='--')
    axs[0].legend(loc='lower right')

    axs[1].set_xlabel('Horizon')
    axs[1].set_ylabel('Number of clicks', color=expColor)
    axs[1].plot(optRewards, optimalColor, label='Optimal Reward')
    axs[1].plot(meanExpectedRewards, expColor, label='Expected Reward')
    axs[1].plot(meanExpectedRewardsSW, expColor, label='Expected Reward SW', linestyle='--')
    axs[1].legend(loc='lower right')

    plt.show()


def PlotCumulativeRegret(logs, categories, candidates, optReward):
    totalRegretList = []
    nExp = len(logs)
    expValueCategories = np.multiply(categories, candidates)
    bestExpValueCategories = optReward

    for e in range(nExp):
        regretList = []
        for logT in logs[e]:
            personType = logT[0]
            proposedPrice = logT[1]
            bestExpValue = bestExpValueCategories[personType]
            expValue = expValueCategories[personType][proposedPrice]
            regretT = bestExpValue - expValue
            regretList.append(regretT)
        totalRegretList.append(regretList)
    avgRegretList = np.cumsum(np.mean(totalRegretList, axis=0))

    rewardsPerExp = []
    for e in range(nExp):

        rewards = []
        for logT in logs[e]:
            reward = logT[2]
            rewards.append(reward)

        rewardsPerExp.append(rewards)
    avgRewards = np.cumsum(np.mean(rewardsPerExp, axis=0))

    regretColor = 'tab:red'
    expColor = 'tab:blue'

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Regret (€)', color=regretColor)
    ax1.plot(avgRegretList, regretColor, label='Cumulative Regret')
    ax1.tick_params(axis='y', labelcolor=regretColor)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Reward (€)', color=expColor)
    ax2.plot(avgRewards, expColor, label='Expected Reward')
    ax2.tick_params(axis='y', labelcolor=expColor)
    ax2.legend(loc='lower right')

    fig.tight_layout()
    plt.show()
