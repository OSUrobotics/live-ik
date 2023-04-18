#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:57:39 2023

@author: orochi
"""
import copy

class State_RW:
    def __init__(self):
        self.data = {'current_state': {'two_finger_gripper': {'pose': [[], []],'joint_angles': {'l_prox_pin': 0.75,'l_distal_pin': -1.4,'r_prox_pin': -0.75, 'r_distal_pin': 1.4}},
         'obj_2': {'pose': [[], []],'velocity': [[], []]},'goal_pose': {'goal_pose': []},
         'f1_pos': [],
         'f2_pos': [],
         'f1_base': [],
         'f2_base': []},
         'previous_state': [{'two_finger_gripper': {'pose': [[], []],
           'joint_angles': {'l_prox_pin': 0.75,
            'l_distal_pin': -1.4,
            'r_prox_pin': -0.75,
            'r_distal_pin': 1.4}},
          'obj_2': {'pose': [[], []],
           'velocity': [[], []]},
          'goal_pose': {'goal_pose': []},
          'f1_pos': [],
          'f2_pos': [],
          'f1_base': [],
          'f2_base': []}*5]}
        self.pflag = False
    
    def set_goal_pose(self, gp):
        self.data['current_state']['goal_pose']['goal_pose'] = copy.deepcopy(gp)
        self.pflag = False
    
    def update_state(self, object_pos, fingertip_pos, fingerbase_pos, joint_angles):
        if self.pflag:
            self.data['previous_state'][1:] = self.data['previous_state'][0:-1]
            self.data['previous_state'][0] = self.data['current_state'].copy()
            
        self.data['current_state']['obj_2']['pose'][0] = copy.deepcopy(object_pos)
        for k in self.data['current_state']['two_finger_gripper']['joint_angles'].keys():
            self.data['current_state']['two_finger_gripper']['joint_angles'][k] = joint_angles[k] # could also assign based on inds
        self.data['current_state']['f1_pos'] = fingertip_pos[0:2]
        self.data['current_state']['f2_pos'] = fingertip_pos[2:4]
        self.data['current_state']['f1_base'] = fingerbase_pos[0:2]
        self.data['current_state']['f2_base'] = fingerbase_pos[2:4]
        
        if not self.pflag:
            self.data['previous_state'] = [self.data['current_state']] * 5
            self.pflag = True
    
    def get_state(self):
        return self.data