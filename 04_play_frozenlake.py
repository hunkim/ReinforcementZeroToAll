import gym
import readchar

import utils.prints as print_utils

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {'\x1b[A': UP, '\x1b[B': DOWN, '\x1b[C': RIGHT, '\x1b[D': LEFT}

# is_slippery True
env = gym.make('FrozenLake-v0')

env.reset()

print_utils.clear_screen()
env.render()  # Show the initial board

while True:
    # Choose an action from keyboard
    key = readchar.readkey()

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)

    # Show the board after action
    print_utils.clear_screen()
    env.render()

    print("State: {} Action: {} Reward: {} Info: {}".format(
        state, action, reward, info))

    if done:
        print_utils.print_result(reward)
