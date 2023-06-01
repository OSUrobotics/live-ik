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
         'previous_state': [[{'two_finger_gripper': {'pose': [[], []],
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
          'f2_base': []}]*5]}
        self.pflag = False
        self.names = ['r_prox_pin', 'r_distal_pin','l_prox_pin','l_distal_pin',]
    
    def set_goal_pose(self, gp):
        self.data['current_state']['goal_pose']['goal_pose'] = copy.deepcopy(gp)
        self.pflag = False
    
    def update_state(self, object_pos, fingertip_pos, fingerbase_pos, joint_angles):
        if self.pflag:
            print("HERE PFLAG IS TRUE")
            self.data['previous_state'][1:] = copy.deepcopy(self.data['previous_state'][0:-1])
            self.data['previous_state'][0] = copy.deepcopy(self.data['current_state'])
            
        self.data['current_state']['obj_2']['pose'][0] = copy.deepcopy(object_pos)
        for k in self.data['current_state']['two_finger_gripper']['joint_angles'].keys():
            self.data['current_state']['two_finger_gripper']['joint_angles'][k] = joint_angles[k] # could also assign based on inds
        self.data['current_state']['f1_pos'] = copy.deepcopy(fingertip_pos[0:2])
        self.data['current_state']['f2_pos'] = copy.deepcopy(fingertip_pos[2:4])
        self.data['current_state']['f1_base'] = copy.deepcopy(fingerbase_pos[0:2])
        self.data['current_state']['f2_base'] = copy.deepcopy(fingerbase_pos[2:4])
        self.data['current_state']['f1_ang'] = self.data['current_state']['two_finger_gripper']['joint_angles']['r_prox_pin'] + self.data['current_state']['two_finger_gripper']['joint_angles']['r_distal_pin']
        self.data['current_state']['f2_ang'] = self.data['current_state']['two_finger_gripper']['joint_angles']['l_prox_pin'] + self.data['current_state']['two_finger_gripper']['joint_angles']['l_distal_pin']
        
        if not self.pflag:
            self.data['previous_state'] = [copy.deepcopy(self.data['current_state']),copy.deepcopy(self.data['current_state']),copy.deepcopy(self.data['current_state']),copy.deepcopy(self.data['current_state']),copy.deepcopy(self.data['current_state'])] 
            self.pflag = True

    
    def get_state(self):
        return self.data