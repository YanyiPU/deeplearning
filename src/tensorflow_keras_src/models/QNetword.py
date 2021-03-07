import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque


"""
env = gym.make("CartPole-v1")
state = env.reset()
while True:
    env.render()
    action = model.predict(state)
    next_state, reward, done, info = env.step(action)
    if done:
        break
"""


num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 32
learning_rate = 1e-3
gamma = 1.
initial_epsilon = 1.
final_epsilon = 0.01


class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = 24, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 24, activation = tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units = 2)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_valuse, axis = -1)





if __name__ == "__main__":
    # 实例化一个游戏环境，参数为游戏名称
    env = gym.make("CartPole-v1")
    
    # Q-Learning 算法
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    replay_buffer = deque(maxlen = 10000) # 使用一个 deque 作为 Q-Learning 的经验回放池

