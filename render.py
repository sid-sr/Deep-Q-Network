''' To test out the DQN implementation by rendering the environment.
'''

from keras.models import model_from_json
import numpy as np
import gym

json_file = open('./ModelFiles/model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights('./ModelFiles/model.h5')

env = gym.envs.make('CartPole-v1')


def test_run(env, model):
    ''' Testing the model on the environment with a greedy policy function.
    '''
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = np.argmax(model.predict(state.reshape(1, -1))[0])
        next_state, reward, done, _ = env.step(action)
        state = next_state


test_run(env, model)
