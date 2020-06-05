import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise():
	def __init__(self, mu, sigma = 0.15, theta = 0.2, delta = 1e-3, x0= None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
		self.x_prev = x 

		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer():
	def __init__(self, max_size, input_shape, n_actions):
		self.memory_size = max_size
		self.memory_counter = 0
		self.state_memory = np.zeros((self.memory_size, *input_shape))
		self.new_state_memory = np.zeros((self.memory_size, *input_shape))
		self.action_memory = np.zeros((self.memory_size, n_actions))
		self.reward_memory = np.zeros(self.memory_size)
		self.terminal_state = np.zeros(self.memory_size, dtype = np.float32)

	def store_transition(self, state, action, reward, state_, done):
		index = self.memory_counter % self.memory_size
		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.new_state_memory[index] = state_
		self.terminal_state[index] = 1 - done
		self.memory_counter += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.memory_counter, self.memory_size)
		batch = np.random.choice(max_mem, batch_size)

		states = self.state_memoryp[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		terminal_states = self.terminal_state[batch]
		new_states = self.new_state_memory[batch]

		return states, actions, rewards, terminal_states, new_states

class CriticNetwork(nn.Module):
	def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = 'tmp/ddpg'):
		super(CriticNetwork, self).__init__()
		self.input_dims = input_dims
		self.n_actions = n_actions
		self.checkpoint_file = os.path.join(chkpt_dir, name +"_ddpg")

		self.fc1 = nn.Linear(*self.input_dims , fc1_dims)
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

		self.bn1 = nn.LayerNorm(self.fc1_dims)

		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		f2 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

		self.bn2 = nn.LayerNorm(self.fc2_dims)

		self.action_value =nn.Linear(self.n_actions, fc2_dims)
		f3 = 0.003
		self.q = nn.Linear(self.fc2_dims, 1)
		T.nn.init.uniform_(self.q.weight.data, -f3, f3)
		T.nn.init.uniform_(self.q.bias.data, -f3, f3)

		self.optimizer = optim.Admin(self.parameters(), lr = beta)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state, actions):
		state_value = self.fc1(state)
		state_value = self.bn1(state_value)
		state_value = F.relu(state_value)
		state_value = self.fc2(state_value)
		state_value = self.bn2(state_value)

		action_value = self.action_value(action)
		state_action_value = F.relu(T.add(state_value, action_value))

		return state_action_value

	def save_checkpoint(self):
		print("Saving checkpoint")
		T.save(self.sate_dict(), self.checkpoint_file)

	def load_checkpoin(self):
		print("Laoding checkpoint")
		self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
	def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = "tmp/dir"):
		super(ActorNetwork, self).__init__()
		self.input_dims = input_dims
		self.n_actions = n_actions
		self.checkpoint_file = os.path.join(chkpt_dir, name +"_ddpg")

		self.fc1 = nn.Linear(*self.input_dims , fc1_dims)
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

		self.bn1 = nn.LayerNorm(self.fc1_dims)

		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		f2 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

		self.bn2 = nn.LayerNorm(self.fc2_dims)

		f3 = 0.003
		self.mu = nn.Linear(fc2_dims, self.n_actions)
		f3 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
		T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

		self.optimizer = optim.Admin(self.parameters(), lr = beta)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state):
		state = self.fc1(state)
		state = self.bn1(state)
		state = F.relu(state)

		state = self.fc2(state)
		state = self.bn2(state)
		state = F.relu(state)
		state = T.tanh(self.mu(state))

		return state


	def save_checkpoint(self):
		print("Saving checkpoint")
		T.save(self.sate_dict(), self.checkpoint_file)

	def load_checkpoin(self):
		print("Laoding checkpoint")
		self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
	def __init__(self, alpha, beta, input_dims, tau, env, gamma = 0.99, n_actions = 2, max_size = 10000000, layer1_size = 400, layer2_size = 300, batch_size = 64):

		self.alpha = alpha
		self.beta = beta
		self.input_shape = input_dims
		self.tau = tau
		self.env = env
		self.gamma = gamma
		self.n_actions = n_actions
		self.max_size = max_size
		self.layer1_size = layer1_size
		self.layer2_size = layer2_size
		self.batch_size = batch_size

		self.memory = ReplayBuffer(max_size, input_dims, n_actions)

		self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = "Actor")
		self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = "TargetActor")

		self.crititc = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = "Critic")
		self.target_crititc = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = "TargetCritic")

		self.noise = OUActionNoise(mu = np.zeros(n_actions))

		self.update_network_parameters(tau=1)

	def choose_action(self, observation):
		self.actor.eval()
		observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
		mu = self.actor(observation).to(self.actor.device)
		mu_prime = mu + T.tensor(self.noise(), dtype.T.float).to(self.actor.device)

		self.actor.train()

		return mu_prime.cpu().detach().numpy()

	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def learn(self):
		if self.memory.memory_counter < self.batch_size:
			return
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		state = T.tensor(state, dtype=T.float).to(self.crititc.device)
		action = T.tensor(action, dtype=T.float).to(self.crititc.device)
		new_state = T.tensor(new_state, dtype=T.float).to(self.crititc.device)
		reward = T.tensor(reward, dtype=T.float).to(self.crititc.device)
		done = T.tensor(done, dtype=T.float).to(self.crititc.device)

		self.target_actor.eval()
		self.target_crititc.eval()
		self.crititc.eval()

		target_action = self.target_actor.forward(new_state)
		critic_value_ = self.target_crititc.forward(new_state, target_action)
		critic_value = self.crititc.forward(state, action)

		target = []
		for j in range(self.batch_size):
			target.append(reward[j] + self.gamma * critic_value_[j] * done[j])

		target = T.tensor(target, dtype=T.float).to(self.crititc.device)
		target = target.view(self.batch_size, 1)

		self.crititc.train()
		self.crititc.optimizer.zero_grad()

		critic_loss = F.mse_loss(target, critic_value)
		critit_loss.backward()
		self.critic.optimizer.step()

		self.critic.eval()
		self.actor.optimizer.zero_grad()

		mu = self.actor.forward
		self.actor.train()
		actor_loss = -self.critic.forward(state, mu)
		actor_loss = T.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()

		self.update_network_parameters()

	def update_network_parameters(self, tau = None):
		if tau is None:
			tau = self.tau

		actor_params = self.actor.named_parameters()
		critic_params = self.critic.named_parameters()

		target_actor = self.target_actor.named_parameters()
		target_critic = self.target_critic.named_parameters()

		critic_state_dict = dict(critic_params)
		actor_state_dict = dict(actor_params)
		target_critic_state_dict = dict(critic_params)
		target_actor_state_dict = dict(target_actor)

		for name in critic_state_dict:
			critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1-tau) ( target_critic_state_dict[name].clone())
		self.target_critic.load_state_dict(critic_state_dict)

		for name in actor_state_dict:
			actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau) ( target_actor_state_dict[name].clone())
		self.target_actor.load_state_dict(actor_state_dict)

	def save_models(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.target_critic.load_checkpoint()