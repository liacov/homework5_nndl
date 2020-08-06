import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import numpy as np



"""
Plot a matrix representing the status of the environment
"""

#produce colormap with as many colors as there are unique values in df
colors = ["white", "red", "grey", "limegreen", "gold", "palegreen", "orange"]
# tiles, holes, walls, agent, goal, trajectory
cmap = ListedColormap(colors)




def plot(size, goal_pos, obstacles, holes, spikes, traj, directory):

    for i, current_pos in enumerate(traj):
        fig, ax = plt.subplots(figsize=(10, 10))

        img = np.zeros((size[0], size[1]))

        img[goal_pos[0], goal_pos[1]] = 4

        for pit in holes:
            img[pit[0], pit[1]] = 1

        for s in spikes:
            img[s[0], s[1]] = 6

        for obs in obstacles:
            img[obs[0], obs[1]] = 2

        agent_pos = current_pos
        img[agent_pos[0], agent_pos[1]] = 3

        ax.text(goal_pos[1], goal_pos[0], "GOAL",
                ha="center", va="center", color="black",
                fontsize = 18)

        ax.text(agent_pos[1], agent_pos[0], "A",
                ha="center", va="bottom", color="green",
                fontsize = 18)

        for obs in obstacles:
            ax.text(obs[1], obs[0], "WALL",
                    ha="center", va="center", color="black",
                    fontsize = 18)

        for pit in holes:
            ax.text(pit[1], pit[0], "HOLE",
                    ha="center", va="top", color="black",
                    fontsize = 18)

        for s in spikes:
            ax.text(s[1], s[0], "SPIKES",
                    ha="center", va="top", color="black",
                    fontsize = 18)

        for j, point in enumerate(traj[:i]):
            img[point[0], point[1]] = 5
            ax.text(point[1], point[0], j,
                    ha="center", va="center", color="b",
                    fontsize = 18)

        im = ax.imshow(img, vmin=0, vmax=len(cmap.colors), cmap=cmap)

        fig.tight_layout()
        plt.savefig(directory+"/{}.png".format(i))
        plt.close()
