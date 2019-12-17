import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.
        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.
        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.num_actions = num_actions
        self.hidden_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Optimizer
        # TODO: Define network parameters and optimizer
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_size, input_shape=(state_size,), use_bias=True,
                                                  activation='relu')  # Actor layer 1
        self.output_layer = tf.keras.layers.Dense(num_actions, input_shape=(self.hidden_size,), use_bias=True,
                                                  activation='softmax')  # Actor layer 2
        self.hidden_critic_layer = tf.keras.layers.Dense(self.hidden_size, input_shape=(state_size,), use_bias=True,
                                                         activation='relu')  # Critic layer 1
        self.output_critic_layer = tf.keras.layers.Dense(1, input_shape=(self.hidden_size,),
                                                         use_bias=True)  # Critic layer 2
        pass

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
        # TODO: implement this!
        hidden_output = self.hidden_layer(states)
        return self.output_layer(hidden_output)
        pass

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.
        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode
        :return: A [episode_length] matrix representing the value of each state
        """
        # TODO: implement this :D
        hidden_critic_output = self.hidden_critic_layer(states)
        return self.output_critic_layer(hidden_critic_output)
        pass

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the handout to see how this is done.
        Remember that the loss is similar to the loss as in reinforce.py, with one specific change.
        1) Instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. Here, advantage is defined as discounted_rewards - state_values, where state_values is calculated by the critic network.

        2) In your actor loss, you must set advantage to be tf.stop_gradient(discounted_rewards - state_values). You may need to cast your (discounted_rewards - state_values) to tf.float32. tf.stop_gradient is used here to stop the loss calculated on the actor network from propagating back to the critic network.

        3) To calculate the loss for your critic network. Do this by calling the value_function on the states and then taking the sum of the squared advantage.
        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        probabilities = self.call(states)  # Gets probabilities from actor
        value = self.value_function(states)  # Gets value
        advantage = discounted_rewards - value  # Advantage
        critic_loss = tf.reduce_sum(advantage ** 2)  # Critic's loss
        indices = list(
            zip(np.arange(probabilities.shape[0]), actions))  # Makes tuples of action index with action number.
        probabilities_for_actual_actions = tf.gather_nd(probabilities,
                                                        indices)  # Gets the probabilities of taken actions.
        negative_log_probs = -tf.math.log(probabilities_for_actual_actions)
        return tf.reduce_sum(negative_log_probs * tf.stop_gradient(advantage)) + 0.5 * critic_loss  # Gets total loss.
        pass