import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.
        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) # Optimizer
        # TODO: Define network parameters and optimizer
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_size, input_shape=(state_size,), use_bias=True, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, input_shape=(self.hidden_size,), use_bias=True, activation='softmax')
    @tf.function
    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.
        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        # TODO: implement this ~
        hidden_output = self.hidden_layer(states)
        return self.output_layer(hidden_output)
        pass

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.
        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this uWu
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        probabilities = self.call(states)
        indices = list(zip(np.arange(probabilities.shape[0]),actions)) # Makes tuples of action index with action number.
        probabilities_for_actual_actions = tf.gather_nd(probabilities,indices) # Gets the probabilities of taken actions.
        negative_log_probs = -tf.math.log(probabilities_for_actual_actions)
        return tf.reduce_sum(negative_log_probs*discounted_rewards) # Gets loss
        pass