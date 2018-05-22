import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split


# Simple Feedforward Neural Network
def simple_nn(obs_size, act_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(obs_size, )))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(act_size, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    #Firstread the expert policies
    f=open('Silvia_expert100.pkl', 'rb')
    d=f.read()
    data = pickle.loads(d)
    obs = np.asarray(data['obs'])

    act_list = data['act']
    len_row = len(act_list)
    len_col = len(act_list[0])
    act = np.zeros([len_row, len_col])
    for i in range(len_row):
       act[i][:] = act_list[i][:]
    
    # This data has 100 rounds, the size of observation is 167, the size of action is 18
    obs_length = len(obs[1])
    act_length = len(act[1])

    obs_train, obs_test, act_train, act_test = train_test_split(obs, act, test_size=0.2, random_state=245323)
    model = simple_nn(obs_length, act_length)
    model.fit(obs_train, act_train, batch_size=64, epochs=1, verbose=1)
    model.save('models/' + 'Silvia' + '.h5py')
