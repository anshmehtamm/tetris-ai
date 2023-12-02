from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import random
from collections import deque
import time
from modified_tensor_board import ModifiedTensorBoard


ACTION_SPACE = 5
REPLAY_MEMORY_SIZE = 100000
MODEL_NAME = "tetris_q_net"
MIN_REPLAY_MEMORY_SIZE = 4000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class TetrisQNet:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        # conv net, input shape is 177x43x1
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(177,43,1)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # dense layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(ACTION_SPACE, activation='linear'))

        # compile model
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, experience):
        self.replay_memory.append(experience)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]
    
    def train(self, terminal_state, step):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # get a random sample from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # get current states from minibatch, then query NN model for Q values
        current_states = np.array([experience[0] for experience in minibatch])/255
        current_qs_list = self.model.predict(current_states, verbose=0)

        # get future states from minibatch, then query NN model for Q values
        # when using target network, query it, otherwise main network should be queried
        new_current_states = np.array([experience[3] for experience in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states,verbose=0 )

        X = []
        y = []

        # enumerate over the minibatch and prepare X and y
        #state, reward, next_state, done
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # if not a terminal state, get new q from future states, otherwise set it to 0
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q

            # update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # append to training data
            X.append(current_state)
            y.append(current_qs)

        # fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # if counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY :
            print("updating target model")
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
        # reset replay memory
        if terminal_state:
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)


    

    


    


    

