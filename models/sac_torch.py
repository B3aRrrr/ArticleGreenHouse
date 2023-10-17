import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math 
import argparse
import os
from utils import plotLearning



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Reward_adapter(
    r,
    #EnvIdex
    ):
    # # For BipedalWalker
    # if EnvIdex == 0 or EnvIdex == 1:
    if r <= -100: r = -1
    # # For Pendulum-v0
    # elif EnvIdex == 3:
    #     r = (r + 8) / 8
    return r
def Done_adapter(r,done,current_steps, EnvIdex):
    # For BipedalWalker
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: Done = True
        else: Done = False
    else:
        Done = done
    return Done

def Action_adapter(a,max_action):
    #from [-1,1] to [-max,max]
    return  a*max_action
def Action_adapter_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return  act/max_action

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')   
def evaluate_policy(env, model, render, steps_per_epoch, max_action):
    scores = 0
    turns = 3#opt.eval_turn
    for j in range(turns):
        s, done, ep_r = env.reset()[0], False, 0
        while not done: 
            a = model.select_action(s, deterministic=True, with_logprob=False)
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            s_prime, r, done, info = env.step(act,False) 
            s = s_prime
            if render:
                env.render()
        # print(ep_r)
        scores += ep_r
    return scores/turns


class RandomBuffer(object):
	def __init__(self, state_dim, action_dim, Env_with_dead , max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.Env_with_dead = Env_with_dead

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dead = np.zeros((max_size, 1),dtype=np.uint8)

		self.device = device


	def add(self, state, action, reward, next_state, dead):
		# print(f'state={state}|self.ptr={self.ptr}')
		if isinstance(state,tuple):self.state[self.ptr] = state[0]
		else:self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		# it is important to distinguish between dead and done!!!
		# See https://zhuanlan.zhihu.com/p/409553262 for better understanding.
		if self.Env_with_dead:
			self.dead[self.ptr] = dead
		else:
			self.dead[self.ptr] = False

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		with torch.no_grad():
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.dead[ind]).to(self.device)
			)

	def save(self):
		'''save the replay buffer if you want'''
		scaller = np.array([self.max_size,self.ptr,self.size,self.Env_with_dead],dtype=np.uint32)
		np.save("buffer/scaller.npy",scaller)
		np.save("buffer/state.npy", self.state)
		np.save("buffer/action.npy", self.action)
		np.save("buffer/reward.npy", self.reward)
		np.save("buffer/next_state.npy", self.next_state)
		np.save("buffer/dead.npy", self.dead)

	def load(self):
		scaller = np.load("buffer/scaller.npy")

		self.max_size = scaller[0]
		self.ptr = scaller[1]
		self.size = scaller[2]
		self.Env_with_dead = scaller[3]

		self.state = np.load("buffer/state.npy")
		self.action = np.load("buffer/action.npy")
		self.reward = np.load("buffer/reward.npy")
		self.next_state = np.load("buffer/next_state.npy")
		self.dead = np.load("buffer/dead.npy")

def build_net(layer_shape, activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor, self).__init__()

        layers = [state_dim] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.device = device;self.to(self.device)

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic: u = mu
        else: u = dist.rsample() #'''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

        self.device = device;self.to(self.device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2
class AgentSAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        hid_shape=(256,256),
        a_lr=3e-4,
        c_lr=3e-4,
        batch_size = 256,
        alpha = 0.2,
        adaptive_alpha = True,
        agent_dir = os.path.join(os.getcwd(),'DefaultAgent')):
            if not os.path.exists(agent_dir):
                # Create a new directory because it does not exist
                os.makedirs(agent_dir)
                print("The new directory is created!")
                
            self.chkpt_dir = agent_dir
            
            self.actor = Actor(state_dim, action_dim, hid_shape).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

            self.q_critic = Q_Critic(state_dim, action_dim, hid_shape).to(device)
            self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
            self.q_critic_target = copy.deepcopy(self.q_critic)
            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.q_critic_target.parameters():
                p.requires_grad = False

            self.action_dim = action_dim
            self.gamma = gamma
            self.tau = 0.005
            self.batch_size = batch_size

            self.alpha = alpha
            self.adaptive_alpha = adaptive_alpha
            if adaptive_alpha:
                # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
                self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
                # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
                self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr) 
            self.score_history=[]
            
    def select_action(self, state, deterministic, with_logprob=False):
        # print(f'[AgentSAC][INPUT] state={state}')
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # print(f'[AgentSAC] state={state}')
            a, _ = self.actor(state, deterministic, with_logprob)
        return a.cpu().numpy().flatten()
    def learn(self,replay_buffer):
        
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime)
            target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (target_Q - self.alpha * log_pi_a_prime) #Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = 	False

        a, log_pi_a = self.actor(s)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = 	True
        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def train(
        self,env,total_steps,
        # EnvIdex,
        write,writer,
        # EnvName=['BipedalWalker-v3','BipedalWalkerHardcore-v3','LunarLanderContinuous-v2','Pendulum-v0','Humanoid-v2','HalfCheetah-v2'],
        update_every:int=50,
        eval_interval:int=int(1e3),
        save_interval:int=int(2.5e3),# opt.save_interval  #in steps
        random_seed:int=42,
        plot:bool=True,
        plot_path:str= os.path.join(os.getcwd(),'plots',"BipedalWalker-v3"),
        isProfile:bool=False):

            if not os.path.exists(plot_path):
                    # Create a new directory because it does not exist
                    os.makedirs(plot_path)
            replay_buffer = RandomBuffer([*env.observation_space.shape][0], env.action_space.shape[0], True, max_size=int(1e6))
            max_action = float(env.action_space.high[0])
            steps_per_epoch = env._max_episode_steps
            #Interaction config:

            write = True # opt.write

            start_steps = 5 * steps_per_epoch #in steps
            update_after = 2 * steps_per_epoch #in steps
            eval_env = env


            s, done, current_steps = env.reset(), False, 0


            # print(f's={s}; done = {done};')
            for t in range(total_steps):
                current_steps += 1
                '''Interact & trian'''

                if t < start_steps:
                    #Random explore for start_steps
                    act = env.action_space.sample() #act∈[-max,max]
                    a = Action_adapter_reverse(act,max_action) #a∈[-1,1]
                else:
                    # print(f'[AgentSAC] total_steps = {total_steps}; [Action_adapter];s={s[0]}; t = {t}')
                    a = self.select_action(s, deterministic=False, with_logprob=False) #a∈[-1,1]
                    act = Action_adapter(a,max_action) #act∈[-max,max]

                s_prime, r, done, info = env.step(act,isProfile)#s_prime, r, done, info = env.step(act)
                dead = Done_adapter(r, done, current_steps)#, EnvIdex)
                r = Reward_adapter(r)#, EnvIdex)
                replay_buffer.add(s, a, r, s_prime, dead)
                s = s_prime


                # 50 environment steps company with 50 gradient steps.
                # Stabler than 1 environment step company with 1 gradient step.
                if t >= update_after and t % update_every == 0:
                    for j in range(update_every):
                        # print(f'[AgentSAC] [LEARNING: STEP] = {j} from {update_every} steps')
                        self.learn(replay_buffer)

                '''save model'''
                if (t + 1) % save_interval == 0:
                    self.save(t + 1)

                '''record & log'''
                if (t + 1) % eval_interval == 0:
                    score = evaluate_policy(eval_env, self, False, steps_per_epoch, max_action)#, EnvIdex)
                    if write:
                        writer.add_scalar('ep_r', score, global_step=t + 1)
                        writer.add_scalar('alpha', self.alpha, global_step=t + 1)
                    print(f'EnvName: {env.__class__.name}; seed: {random_seed}; totalsteps: {t+1}, score: {score}')
                if done:
                    s, done, current_steps = env.reset(), False, 0
                self.score_history.append(r)
                if plot:
                     if (t + 1) % save_interval == 0:
                        filename = f'{t+1}.png'
                        plotLearning(self.score_history,os.path.join(plot_path,filename),window=100)


            env.close()
            eval_env.close()
    def save(self,episode):
        print('...saving checkpoint...')
        if not os.path.exists(os.path.join(self.chkpt_dir,'model')):
                # Create a new directory because it does not exist
                os.makedirs(os.path.join(self.chkpt_dir,'model'))
                print("The new directory is created!")
        torch.save(self.actor.state_dict(), os.path.join(self.chkpt_dir,'model',f'sac_actor{episode}.pth'))
        torch.save(self.q_critic.state_dict(),  os.path.join(self.chkpt_dir,'model',f'sac_q_critic{episode}.pth'))#f"{self.chkpt_dir}/model/sac_q_critic{episode}.pth")
    def load(self,episode):
        
        print('...loading checkpoint...')
        self.actor.load_state_dict(torch.load(os.path.join(self.chkpt_dir,'model',f'sac_actor{episode}.pth')))#f"{self.chkpt_dir}/model/sac_actor{episode}.pth"))
        self.q_critic.load_state_dict(torch.load(os.path.join(self.chkpt_dir,'model',f'sac_q_critic{episode}.pth')))#load_state_dict(torch.load(f"{self.chkpt_dir}/model/sac_q_critic{episode}.pth"))









