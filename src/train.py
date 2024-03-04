from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from tqdm import trange
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
print("Environment created")

state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def reset_memory(self):
        self.data = []
        self.index = 0
    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    

class DQN(nn.Module):
    def __init__(self, input_dim = state_space_dim, 
                 output_dim = action_space_dim, 
                 hidden_dim = 512, 
                 hidden_layers = 5):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.relu = nn.ReLU()
        for i in range(hidden_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return x
    

agent_config = {'nb_actions': env.action_space.n,
          'gamma': 0.98,
          'batch_size': 1024,
          'buffer_size': 100000,
          'epsilon_max': 1.,
          'epsilon_min': 0.01,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 1000,
          'criterion': torch.nn.SmoothL1Loss(),
          'learning_rate': 0.001,
          'gradient_steps': 4,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 20,
          'update_target_tau': 0.005,
          'monitoring_nb_trials': 5,
          'monitoring': 'MCTR',
          'warm_start': False,
          'DQN_hidden_layers': 5,
          'DQN_hidden_dim': 512}

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ProjectAgent:

    def __init__(self, config=agent_config, model=DQN()):
    # def __init__(self):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e6)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitoring = config['monitoring'] if 'monitoring' in config.keys() else 'MCTR'
        self.warmstart = config['warm_start'] if 'warm_start' in config.keys() else False

    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
    
    
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path = 'best_model.pt'):
        model = self.model
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.eval()

    def warming(self, path = 'best_model.pt'):
        model = self.model
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    
    # def buffer_insertion(self, buffer, nb_samples):
    #     state, _ = env.reset()
    #     for _ in trange(nb_samples):
    #         action = env.action_space.sample()
    #         next_state, reward, done, trunc, _ = env.step(action)
    #         buffer.append(state, action, reward, next_state, done)
    
    def train(self, env, max_episode, change_env=False, env_randomization=0.2):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        saving_threshold = 0

        if self.warmstart:
            self.warming()
            print("Model warmed up")

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    #V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    #V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          #", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                    
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                if self.monitoring == 'MCTR' and MC_tr > saving_threshold:
                    self.save('best_model.pt')
                    print("Improved model saved")
                    saving_threshold = MC_tr
                if self.monitoring == 'MCDR' and MC_dr > saving_threshold:
                    self.save('best_model.pt')
                    print("Improved model saved")
                    saving_threshold = MC_dr
                if self.monitoring == 'EPCR' and episode_cum_reward > saving_threshold:
                    self.save('best_model.pt')
                    print("Improved model saved")
                    saving_threshold = episode_cum_reward

                if len(episode_return) % 5 == 0:
                    self.save('model.pt')
                    print("Model saved")

                if change_env:
                    if np.random.rand() < env_randomization:
                        env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
                        # self.memory.reset_memory()
                        print("Environment changed")
                    else:
                        env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

                wandb.log({"Episode return": episode_cum_reward,
                            "MC total reward": MC_tr,
                            "MC discounted reward": MC_dr})
                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    





if __name__ == "__main__":

    agent_config = {'nb_actions': env.action_space.n,
          'gamma': 0.98,
          'batch_size': 1024,
          'buffer_size': 100000,
          'epsilon_max': 1.,
          'epsilon_min': 0.01,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 1000,
          'criterion': torch.nn.SmoothL1Loss(),
          'learning_rate': 0.001,
          'gradient_steps': 4,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 20,
          'update_target_tau': 0.005,
          'monitoring_nb_trials': 5,
          'monitoring': 'MCTR',
          'warm_start': False,
          'DQN_hidden_layers': 5,
          'DQN_hidden_dim': 512}
    
    model = DQN(input_dim = state_space_dim,
                output_dim = action_space_dim,
                hidden_dim = agent_config['DQN_hidden_dim'],
                hidden_layers = agent_config['DQN_hidden_layers']).to(device)
    
    run = wandb.init(project = "rl-class-assignment",
                     config = agent_config)
                     
                     
    
    print("Model created")

    agent = ProjectAgent(config=agent_config, model=model)

    print("Agent created")
    print("Training...")
    episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state = agent.train(env, 1000, change_env=True)





            
            
            
            
            
            
            
            
            
            
            