import numpy as np
import reinforcement_lib
from reinforcement_lib import DQN, Memory
import gym, sys

# simulation control variables
episode = 100
epsilon = 0.1
gamma = 0.98
interval = 200
batch = 32

# creating evironment
env = gym.make("LunarLander-v2")
num_states = 8
num_actions = 1
network_layers = [200,200]
network_name = sys.argv[1] # input('Type the model name:')
#network_name = 'Lunar_01'
target_name = 'Target'

"""
Permissible Actions
0- Do nothing
1- Fire left engine
2- Fire down engine
3- Fire right engine
"""
actions = [0,1,2,3]

# system objects
network = DQN(num_states,num_actions,network_layers, model_name=network_name)
target = DQN(num_states,num_actions,network_layers, model_name=target_name)
memory = Memory(num_states, num_actions)

# system counter
n = 0
curve = np.zeros(episode)

print('Model name:',network_name,'\nEnv States:',num_states,'\nEnv Actions:',num_actions)
print('System read!\nStarting Training...')

# creating training loop
for i in range(episode):
    observation = env.reset()

    done = False

    while not done:

        n+=1
        # render environment
        if True:#i%episode == episode -2:
            env.render()

        # selection action using network
        action = actions[np.argmax(network(observation,actions))]

        # Selection action with epsilon-Greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)

        # saving previous observation
        prev_obs = observation
        observation, reward, done, info = env.step(action)

        # saving transition to the replay memory
        curve[i] += reward
        memory.add(prev_obs,action, reward, observation, done)

        # updating network training
        if len(memory) > 1000:
            bs, ba, br, bsp, bd = memory.sample(batch)

            qsp = np.amax(target(bsp, actions), axis=1, keepdims=True)
            y = br + (1 - bd) * gamma * qsp

            network.train(bs, ba, y)

            # update target network every <interval> steps into training cicle
            if n % interval == 0:
                target <<= network

    # saving the model evey 10 training epochs
    if i%10 == 0:
        network.save_model()

    print('%', i, curve[i])

print('Training done!')
