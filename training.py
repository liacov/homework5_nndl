import dill
import argparse
import numpy as np
import agent
import environment
import plot_world

from pathlib import Path

import pickle


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the agent.')

parser.add_argument('--softmax', action='store_true', help='Enable softmax over e-greedy')
parser.add_argument('--sarsa', action='store_true', help='Enable SARSA over Q-learning')
parser.add_argument('--no_plot', action='store_true', help='Plot the trajectory')
parser.add_argument('--holes', action='store_true', help='Activate holes')
parser.add_argument('--spikes', action='store_true', help='Activate spikes')
parser.add_argument('--discount', type=float, default=0.9, help='Set discount factor (positive real value)')

# parse input arguments
args = parser.parse_args()

print(args)

# set plot flag
PLOT = True
if args.no_plot: PLOT = False

episodes = 2000         # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [0, 2]           # objective point

# save walls positions
obstacles = [[2,0], [2,1], [2,2], [2,3], [2,4], [1,4],
            [0, 6], [1,6], [2,6], [3,6], [4,6], [5,6], [6,6]]

# set additional obstacles, if required
holes = []
spikes = []

if args.holes and args.spikes:
    holes = [[6,7], [7,6], [7,7], [7,5], [6,5], [5,5], [4,5],
             [7,3], [6,3], [5,3]]
    spikes = [[7,4], [6,4], [5,4]]

    elements = 'holes&spikes'

elif args.holes:
    holes = [[6,7], [7,6], [7,7], [7,5], [6,5], [5,5], [4,5],
             [7,3], [6,3], [5,3]]
    elements = 'holes'
elif args.spikes:
    spikes = [[6,7], [7,6],
              [7,5], [6,5], [5,5], [4,5],
              [7,4], [6,4], [5,4],
              [7,3], [6,3]]
    elements = 'spikes'

else: elements = 'walls'

# set hyperparametes
discount = args.discount          # exponential discount factor
softmax = args.softmax         # set to true to use Softmax policy
sarsa = args.sarsa           # set to true to use the Sarsa algorithm

# alpha and epsilon profile
alpha = np.ones(episodes) * 0.3
# linear decay
# epsilon = np.linspace(0.8, 0.001,episodes)
# hyperbolic decay
epsilon = np.linspace(1, 100, episodes)
epsilon = 1/epsilon

# initialize the agent
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

reward_log = []

# perform the training
for index in range(0, episodes):
    # start from the bottom right
    initial = [8,8]

    # start from a random state (not in a hole or wall)
    # initial = [np.random.randint(0, x), np.random.randint(0, y)]
    # while (list(initial) in holes) or (list(initial) in obstacles):
    #     initial = [np.random.randint(0, x), np.random.randint(0, y)]
    trajectory = [initial]

    # initialize environment
    state = initial
    env = environment.Environment(x, y, obstacles, holes, spikes, state, goal)
    reward = 0
    # run episode
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
        result = env.move(action)
        # Q-learning update
        next_index = result[0][0] * y + result[0][1]
        learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
        # update state and reward
        reward += result[1]
        state = result[0]
        trajectory.append([state[0], state[1]])
    reward /= episode_length
    reward_log.append(reward)

    # set labels
    if softmax: policy = 'softmax'
    else: policy = 'greedy'
    if sarsa: algorithm = 'sarsa'
    else: algorithm = 'q-learning'

    # periodically save the agent
    if ((index + 1) % 100 == 0):
        with open('agent.obj', 'wb') as agent_file:
            dill.dump(agent, agent_file)
        print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial)
    if ((index  == 1999)) and (PLOT):
        directory = "plots/{}/plots_{}_{}_{}/{}".format(elements, policy, algorithm, discount, index + 1)
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        # save single frames
        plot_world.plot([x,y], goal, obstacles, holes, spikes, trajectory, directory)
        # save animated gifs
        plot_world.make_gif(elements, policy, algorithm, discount, index + 1)
    # dump reward
    with open('rewards/reward_log_{}_{}_{}_{}.pickle'.format(elements, policy, algorithm, discount), 'wb') as log:
            pickle.dump(reward_log, log)
