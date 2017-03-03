# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap
import gym
import random
from gym.envs.registration import register
import matplotlib.pyplot as plt

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v0')
env.render()

num_episodes = 2000

rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    env.reset()
    rAll = 0
    done = False

    while not done:
        # Random action
        action = random.randint(0, env.action_space.n - 1)

        # Get new state and reward from environment
        _state, reward, done, _info = env.step(action)

        # rAll will be 1 if success, o otherwise
        rAll += reward

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
plt.plot(rList)
plt.show()
