import gym
import minerl

env = gym.make('MineRLObtainDiamond-v0')

# set the environment to allow interactive connections on port 6666
# and slow the tick speed to 6666.
env.make_interactive(port=6666, realtime=True)

# reset the env

env.reset()
done = False
# interact as normal.
while not done:
    print('?')
    env.render()