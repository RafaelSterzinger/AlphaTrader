import os
import sys
import gym
import gym_anytrading
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from baseline import ReinforceWithBaseline


def visualize_data(total_rewards):
	"""
	Takes in array of rewards from each episode, visualizes reward over episodes.
	:param rewards: List of rewards from all episodes
	"""

	x_values = arange(0, len(total_rewards), 1)
	y_values = total_rewards
	plot(x_values, y_values)
	xlabel('episodes')
	ylabel('cumulative rewards')
	title('Reward by Episode')
	grid(True)
	show()


def discount(rewards, discount_factor=.99):
	"""
	Takes in a list of rewards for each timestep in an episode,
	and returns a list of the sum of discounted rewards for
	each timestep. Refer to the slides to see how this is done.
	:param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
	:param discount_factor: Gamma discounting factor to use, defaults to .99
	:return: discounted_rewards: list containing the sum of discounted rewards for each timestep in the original
	rewards list
	"""
	# TODO: Compute discounted rewards
	discounted_rewards = rewards.copy()
	for i in range(len(rewards)-2, -1, -1):
		discounted_rewards[i]=discount_factor*discounted_rewards[i+1]+rewards[i] # Discounts rewards in future and adds it to present reward
	return discounted_rewards
def generate_trajectory(env, model):
	"""
	Generates lists of states, actions, and rewards for one complete episode.
	:param env: The openai gym environment
	:param model: The model used to generate the actions
	:return: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps
	in the episode
	"""
	states = []
	actions = []
	rewards = []
	state = env.reset()
	done = False
	info = 1
	while not done:
		# TODO:
		# 1) use model to generate probability distribution over next actions
		# 2) sample from this distribution to pick the next action
		states.append(np.reshape(state,[-1]))
		next_actions = model.call(tf.convert_to_tensor([np.reshape(state,[-1])],dtype=tf.float32)) # Gets the probabilities for the next actions
		action = np.random.choice(a=env.action_space.n,p=np.reshape(next_actions, [-1])) # Picks an action at random
		actions.append(action)
		state, rwd, done, info = env.step(action) # Uses this action
		rewards.append(rwd)
	print("Information:",info)
	print("Max profit:",env.max_possible_profit())
	return states, actions, rewards


def train(env, model):
	"""
	This function should train your model for one episode.
	Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
	and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
	Make sure to return the total reward for the episode.
	:param env: The openai gym environment
	:param model: The model
	:return: The total reward for the episode
	"""

	# TODO:
	# 1) Use generate trajectory to run an episode and get states, actions, and rewards.
	# 2) Compute discounted rewards.
	# 3) Compute the loss from the model and run backpropagation on the model.
	states, actions, rewards = generate_trajectory(env, model) # Get states, actions, and rewards from one iteration
	with tf.GradientTape() as tape:
		loss = model.loss(tf.cast(tf.convert_to_tensor(states),dtype=tf.float32), tf.convert_to_tensor(actions), tf.cast(tf.convert_to_tensor(discount(rewards)),dtype=tf.float32))
	# Gets the gradients for this batch
	gradients = tape.gradient(loss, model.trainable_variables)

	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Does gradient descent
	reward_sum = 0
	for i in rewards:
		reward_sum+=i # Adds rewards
	return reward_sum

def main():
	if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
		print("USAGE: python assignment.py <Model Type>")
		print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
		exit()

	env = gym.make("forex-v0") # environment
	state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
	num_actions = env.action_space.n

	# Initialize model
	if sys.argv[1] == "REINFORCE":
		model = Reinforce(state_size, num_actions)
	elif sys.argv[1] == "REINFORCE_BASELINE":
		model = ReinforceWithBaseline(state_size, num_actions)

	# TODO:
	# 1) Train your model for 650 episodes, passing in the environment and the agent.
	# 2) Append the total reward of the episode into a list keeping track of all of the rewards.
	# 3) After training, print the average of the last 50 rewards you've collected.
	rewards = []
	for i in range(650):
		print(i)
		reward = train(env, model)
		rewards.append(reward)
	print("Average of last 50 rewards:",tf.reduce_mean(rewards[-50:])) # Prints average of final 50 rewards
	# TODO: Visualize your rewards.
	visualize_data(rewards)

if __name__ == '__main__':
	main()