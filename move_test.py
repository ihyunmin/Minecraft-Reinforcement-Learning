import minerl
import gym
from minerl.data import BufferedBatchIter
import cv2


env = gym.make('MineRLTreechop-v0')
# data = minerl.data.make('MineRLObtainDiamond-v0')
# iterator = BufferedBatchIter(data)

obs  = env.reset()
done = False
net_reward = 0

check = 0
while not done:
    env.render()
    action = env.action_space.noop()

    action['camera'] = [0.00, 0.02]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(action)

    print(type(obs["pov"]), obs["pov"].shape)

    
    
    if check >10:
        break
    else:
        cv2.imwrite('test{}.jpg'.format(str(check)), obs["pov"])
        check +=1 
    net_reward += reward

    # print("Total reward: ", net_reward)