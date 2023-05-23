#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:35:52 2023

@author: orochi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pickle as pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def simple_normalize(x_tensor, mins, maxes):
    """
    normalizes a numpy array to -1 and 1 using provided maximums and minimums
    :param x_tensor: - array to be normalized
    :param mins: - array containing minimum values for the parameters in x_tensor
    :param maxes: - array containing maximum values for the parameters in x_tensor
    """
    return ((x_tensor-mins)/(maxes-mins)-0.5) *2
    

def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr


class Actor(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, max_action:float, state_mins:list, state_maxes:list):
        """
        Constructor initializes actor network with input dimension 'state_dim' 
        and output dimension 'action_dim'. State mins and maxes saved for normalization
        """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, action_dim)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.MAXES = torch.tensor(state_maxes, device=device)
        self.MINS = torch.tensor(state_mins, device=device)
        self.max_action = max_action

    def forward(self, state:torch.Tensor):
        """
        Runs state through actor network to get action associated with state
        """
        state = simple_normalize(state, self.MINS, self.MAXES)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, state_mins:list, state_maxes:list):
        """
        Constructor initializes actor network with input dimension 'state_dim'+'action_dim' 
        and output dimension 1. State mins and maxes saved for normalization
        """
        super(Critic, self).__init__()
        self.leaky = nn.LeakyReLU()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.MAXES = torch.tensor(state_maxes, device=device)
        self.MINS = torch.tensor(state_mins, device=device)
        self.max_q_value = 1

    def forward(self, state:torch.Tensor, action:torch.Tensor):
        """
        Concatenates state and action and runs through critic network to return
        q-value associated with state-action pair
        """
        state = simple_normalize(state, self.MINS, self.MAXES)
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q# * self.max_q_value

class DDPGfD_RW():
    def __init__(self, arg_dict: dict):
        """
        Constructor initializes the actor and critic network and use an argument
        dictionary to initialize hyperparameters
        """
        self.STATE_DIM = arg_dict['state_dim']
        self.REWARD_TYPE = arg_dict['reward']
        self.SAMPLING_STRATEGY = arg_dict['sampling']
        self.LAMBDA_1 = arg_dict['rollout_weight']
        self.ROLLOUT_SIZE = arg_dict['rollout_size']
        print('Saving to tensorboard file', arg_dict['tname'])
        self.ACTION_DIM = arg_dict['action_dim']
        self.actor = Actor(self.STATE_DIM, self.ACTION_DIM, arg_dict['max_action'], arg_dict['state_mins'], arg_dict['state_maxes']).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=arg_dict['learning_rate'], weight_decay=1e-4)

        self.critic = Critic(self.STATE_DIM, self.ACTION_DIM, arg_dict['state_mins'], arg_dict['state_maxes']).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=arg_dict['learning_rate'], weight_decay=1e-4)
        
        self.PREV_VALS = arg_dict['pv']
        self.actor_loss = []
        self.critic_loss = []
        self.critic_L1loss = []
        self.critic_LNloss = []
            
        if 'HER' in arg_dict['model']:
            self.USE_HER = True
        else:
            self.USE_HER = False

        self.DISCOUNT = arg_dict['discount']
        self.TAU = arg_dict['tau']
        self.NETWORK_REPL_FREQ = 2
        self.total_it = 0
        self.LOOKBACK_SIZE = 4 # TODO, make this a hyperparameter in the config file

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.BATCH_SIZE = arg_dict['batch_size']
        self.ROLLOUT = True
        self.u_count = 0
        
        self.actor_component = 100000
        self.critic_component = 10
        self.state_list = arg_dict['state_list']
        self.sampled_positions = np.zeros((100,100))
        self.position_xlims = np.linspace(-0.1, 0.1,100)
        self.position_ylims = np.linspace(0.06,0.26,100)
        self.sampled_file = arg_dict['save_path'] + 'sampled_positions.pkl'
        
        try:
            self.SUCCESS_THRESHOLD = arg_dict['sr']/1000
        except KeyError:
            self.SUCCESS_THRESHOLD = 0.002

    def select_action(self, state):
        """
        Method takes in a State object
        Runs state through actor network to get action from policy in numpy array

        :param state: :func:`~mojograsp.simcore.state.State` object.
        :type state: :func:`~mojograsp.simcore.state.State`
        """
        lstate = self.build_state(state)
        lstate = torch.FloatTensor(np.reshape(lstate, (1, -1))).to(device)
        action = self.actor(lstate).cpu().data.numpy().flatten()
        return action

    def grade_action(self, state, action: np.ndarray):
        """
        Method takes in a State object and numpy array containing a policy action
        Runs state and action through critic network to return state-action
        q-value (as float) and gradient of q-value relative to the action (as numpy array)

        :param state: :func:`~mojograsp.simcore.state.State` object.
        :param action: :func:`~np.ndarray` containing action
        :type state: :func:`~mojograsp.simcore.state.State`
        :type action: :func:`~np.ndarray` 
        """
        lstate = self.build_state(state)
        lstate = torch.FloatTensor(np.reshape(lstate, (1, -1))).to(device)
        action = torch.tensor(np.reshape(action, (1,-1)), dtype=float, requires_grad=True, device=device)
        action=action.float()
        action.retain_grad()
        g = self.critic(lstate, action)
        g.backward()
        grade = g.cpu().data.numpy().flatten()
        
        return grade, action.grad.cpu().data.numpy()
    
    def copy(self, policy_to_copy_from):
        """ Copy input policy to be set to another policy instance
		policy_to_copy_from: policy that will be copied from
        """
        # Copy the actor and critic networks
        self.actor = copy.deepcopy(policy_to_copy_from.actor)
        self.actor_target = copy.deepcopy(policy_to_copy_from.actor_target)
        self.actor_optimizer = copy.deepcopy(policy_to_copy_from.actor_optimizer)

        self.critic = copy.deepcopy(policy_to_copy_from.critic)
        self.critic_target = copy.deepcopy(policy_to_copy_from.critic_target)
        self.critic_optimizer = copy.deepcopy(policy_to_copy_from.critic_optimizer)
        our_dir = vars(self)
        for key,value in vars(policy_to_copy_from).items():
            if key.isupper():
                our_dir[key] = value
        # self.DISCOUNT = policy_to_copy_from.DISCOUNT
        # self.TAU = policy_to_copy_from.TAU
        # self.ROLLOUT_SIZE = policy_to_copy_from.ROLLOUT_SIZE
        # self.NETWORK_REPL_FREQ = policy_to_copy_from.NETWORK_REPL_FREQ
        self.total_it = policy_to_copy_from.total_it
        self.avg_evaluation_reward = policy_to_copy_from.avg_evaluation_reward

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.sampling_decay_rate = policy_to_copy_from.sampling_decay_rate
        self.sampling_decay_freq = policy_to_copy_from.sampling_decay_freq
        # self.BATCH_SIZE = policy_to_copy_from.BATCH_SIZE

    def save(self, filename):
        """ Save current policy to given filename
		filename: filename to save policy to
        """
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        np.save(filename + "avg_evaluation_reward", np.array([self.avg_evaluation_reward]))

    def save_sampling(self):
        with open(self.sampled_file, 'wb') as f_obj:
            # print(self.sampled_positions)
            pkl.dump(self.sampled_positions, f_obj)
            
    def load(self, filename):
        """ Load input policy from given filename
		filename: filename to load policy from
        """
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target", map_location=device)) 
        self.actor_target.load_state_dict(torch.load(filename + "_actor_target", map_location=device)) 

        # self.critic_target = copy.deepcopy(self.critic)
        # self.actor_target = copy.deepcopy(self.actor)

    def update_target(self):
        """ Update frozen target networks to be closer to current networks
        """
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def build_state(self, state_container):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                    elif key == 'ftp':
                        state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                    elif key == 'ja':
                        temp = state_container['previous_state'][i]['two_finger_gripper']['joint_angles']
                        state.extend([temp['r_prox_pin'],temp['r_distal_pin'],temp['l_prox_pin'],temp['l_distal_pin']])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_pose'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['current_state']['obj_2']['pose'][0][0:2])
            elif key == 'ftp':
                state.extend(state_container['current_state']['f1_pos'][0:2])
                state.extend(state_container['current_state']['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(state_container['current_state']['f1_base'][0:2])
                state.extend(state_container['current_state']['f2_base'][0:2])
            elif key == 'ja':
                temp = state_container['current_state']['two_finger_gripper']['joint_angles']
                state.extend([temp['r_prox_pin'],temp['r_distal_pin'],temp['l_prox_pin'],temp['l_distal_pin']])
                # print(state)
            elif key == 'gp':
                state.extend(state_container['current_state']['goal_pose']['goal_pose'])
            else:
                raise Exception('key does not match list of known keys')
        return state

    def build_reward(self, reward_container):
        """
        Method takes in a Reward object
        Extracts reward information from state_container and returns it as a float
        based on the reward structure contained in self.REWARD_TYPE

        :param state: :func:`~mojograsp.simcore.reward.Reward` object.
        :type state: :func:`~mojograsp.simcore.reward.Reward`
        """
        if self.REWARD_TYPE == 'Sparse':
            tstep_reward = reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD
        elif self.REWARD_TYPE == 'Distance':
            tstep_reward = max(-reward_container['distance_to_goal'],-1)
        elif self.REWARD_TYPE == 'Distance + Finger':
            tstep_reward = max(-reward_container['distance_to_goal'] - max(reward_container['f1_dist'],reward_container['f2_dist'])/5,-1)
        elif self.REWARD_TYPE == 'Hinge Distance + Finger':
            tstep_reward = reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD + max(-reward_container['distance_to_goal'] - max(reward_container['f1_dist'],reward_container['f2_dist'])/5,-1)
        else:
            raise Exception('reward type does not match list of known reward types')
        return float(tstep_reward)
