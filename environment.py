import numpy as np

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    
    def __init__(self, x, y, obstacles, holes, spikes, initial, goal):
        self.boundary = np.asarray([x, y])
        self.obstacles = obstacles
        self.holes = holes
        self.spikes = spikes
        self.state = np.asarray(initial)
        self.goal = goal
        self.in_the_hole = False
    
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = 0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if (self.in_the_hole):
            reward = -10
        elif(self.check_boundaries(next_state)):
            reward = -1
        elif (self.check_obstacles(next_state)):
            reward = -1
        elif (self.check_spikes(next_state)):
            reward = -0.5
            self.state = next_state
        elif (self.check_holes(next_state)):
            reward = -10
            self.state = next_state
            self.in_the_hole = True
        else:
            self.state = next_state
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    # check if the agent hit an internal wall
    def check_obstacles(self, state):
        return list(state) in self.obstacles

    # check if the agent fell in the hole
    def check_holes(self, state):
        return list(state) in self.holes

    # check if the agent is on the spikes
    def check_spikes(self, state):
        return list(state) in self.spikes
