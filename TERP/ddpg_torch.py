import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks_with_attention import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha, beta, tau, n_actions, gamma=0.99,
                 max_size=10000, batch_size=12):
        self.input_dims = (2,) #dimension of the input states
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, self.input_dims, n_actions)

        self.actor = ActorNetwork(alpha, n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta,  n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta,  n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval() # to keep track of layer statistics
        state1 = T.tensor(observation[0], dtype=T.float).to(self.actor.device) #observation numpy array into a tensor
        state2 = T.tensor(observation[1], dtype=T.float).to(self.actor.device)
        # mu = self.actor.forward(state1.unsqueeze(0).unsqueeze(0),state2.unsqueeze(0)).to(self.actor.device)
        mu, cbam_out, cbam_in  = self.actor.forward(state1.unsqueeze(0).unsqueeze(0),state2.unsqueeze(0)) #.to(self.actor.device)
        mu = mu.to(self.actor.device)
        cbam_out = cbam_out.to(self.actor.device)
        cbam_in = cbam_in.to(self.actor.device)
        mu_prime = mu
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0] , cbam_out.cpu().detach().numpy(), cbam_in.cpu().detach().numpy()

    def remember(self, state1, state2, action, reward, state1_, state2_, done):
        self.memory.store_transition(state1,state2, action, reward, state1_,state2_,done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):

        # if the agent hasn,t filled the least batch size number of transitions in its memory,
        # just return and go back to the main loop
        if self.memory.mem_cntr < self.batch_size:
            return

        states1, states2, actions, rewards, states1_,states2_, done = self.memory.sample_buffer(self.batch_size)

        states1 = T.tensor(states1, dtype=T.float).to(self.actor.device)
        states2 = T.tensor(states2, dtype=T.float).to(self.actor.device)
        states1_ = T.tensor(states1_, dtype=T.float).to(self.actor.device)
        states2_ = T.tensor(states2_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        #passing tensors through the networks
        target_actions, target_cbam_out, target_cbam_in = self.target_actor.forward(states1_,states2_)
        critic_value_ = self.target_critic.forward(states1_,states2_, target_actions)
        critic_value = self.critic.forward(states1,states2, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value) # MSE between the target and critic values
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        mu, cbam_out, cbam_in = self.actor.forward(states1,states2)
        actor_loss = -self.critic.forward(states1,states2, mu )
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        #load state dictionaries into the targer networks
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)
