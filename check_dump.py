#!/usr/bin/env python
# coding: utf-8

# In[38]:


import torch
import torchtext
from torchtext.data import Field
import pandas as pd
import numpy as np
import re
import os
import json

import argparse


# useful fnc for DataFrame

# In[2]:


def get_length_column(df, col):
    return df[col].apply(lambda x: len(x.split()))


# basic process of .csv DataFrame

# In[3]:


def src_cut(s):
    start = s.index('<sos>')+len('<sos>')
    end = s.index('<eos>')
    return s[start:end].strip()


# In[4]:


def conjunct(df_dev, t2u, u2t):
    t2u_join = pd.merge(t2u, df_dev, on='traj')
    t2u_join.loc[:, 'tgt_len'] = get_length_column(t2u_join, 'uttr')
    print(t2u_join.columns)
    
    u2t_join = pd.merge(u2t, df_dev, on='uttr')
    u2t_join.loc[:, 'tgt_len'] = get_length_column(u2t_join, 'traj')
    print(u2t_join.columns)
    
    return t2u_join, u2t_join


# In[15]:


def out_cut(s):
    # cut <sos> and <eos> in out
    start = s.index('<sos>')+len('<sos>')
    end = s.find('<eos>')
    if end == -1:
        return s[start:].strip(), False
    else:
        return s[start:end].strip(), True


# In[16]:


# 'length_flag': -1--no eos; 0--has eos but wrong length; 1--correct length;
def check_length(seri):
    tgt_len = seri['tgt_len']
    flag = -1
    out, haseos = out_cut(seri['out'])
    if haseos:
        flag = 0
        if len(out.split()) == tgt_len:
            flag = 1
    return out, flag


# In[17]:


# markers = ('M1', 'M2', 'M3', 'M4', 'M5')
def get_markers(s, markers):
    pattern = []
    for word in s.split():
        if word in markers:
            pattern.append(word)
    return pattern


# In[31]:


# forward_order = [['M1'], ['M1','M2'], ['M1','M2','M3'], ['M1','M2','M3','M4'], ['M1','M2','M3','M4','M5']]

# further explore --should be 22
# 22 = c50(1) + c51(5) + c52(10) + c53(10) + c54(5) + c55(1)
# forward_potential_order = [['M2'], ['M2','M3'], ['M2','M3','M4'], ['M2','M3','M4','M5'],
#                            ['M3'], ['M3','M4'], ['M3','M4','M5'],
#                            ['M4'], ['M4','M5'], ['M5']]
# 'marker_flag': -2--none; -1--potential pattern; 0--correct pattern; 1--correct marker;
def check_marker(s, tgt, markers, order, potential_order):
    out_marker = get_markers(s, markers)
    tgt_marker = get_markers(tgt, markers)
    if out_marker == tgt_marker:
        flag = 1
    elif out_marker in order:
        flag = 0
    elif out_marker in potential_order:
        flag = -1
    else:
        flag = -2
    return flag


# In[19]:


# further explore
# vocab_cat = {'command': ['left','right','up','down'],
#             'marker': ['M1', 'M2', 'M3', 'M4', 'M5'],
#             'steps': ['1', '2', '3']}

def check_structure(s, vocab_cat):
    seqs = s.split()
    flag = -1
    # for i in range(len(seqs)):
    return flag    
        


# In[28]:


def check_seq(seri, markers, order, potential_order, vocab_cat):
    # ATTENTION: CHANGE LATER to'tgt'
    flags = dict()
    s = seri['out']
    tgt = seri['uttr']
    
    out_, length_flag = check_length(seri)
    flags['length_flag'] = length_flag
    
    marker_flag = check_marker(out_, tgt, markers, order, potential_order)
    flags['marker_flag'] = marker_flag
    
    structure_flag = check_structure(out_, vocab_cat)
    flags['structure_flag'] = structure_flag
    
    return flags


# In[ ]:


# def u2t_acc(u2t_join):
#     return 0

# def t2u_acc(t2u_join):
#     return 0


# functions to get files

# In[5]:


def get_csv_file(csv_dir):
    f = []
    for dir_path, subpaths, files in os.walk(csv_dir):
        f.append((dir_path, files))
    dir_, files = f[0]
    
    f_csv=[]
    for f in files:
        (fname, ext) = os.path.splitext(f)
        if ext == '.csv':
            m = os.path.join(dir_, f)
            f_csv.append(m)
    return f_csv


# In[6]:


def get_iter(f_csv):
    itr = os.path.splitext(os.path.splitext(f_csv)[0])[1]
    return itr


# In[7]:


def get_jsonstat(f_csv):
    # read dump stat
    with open(f_csv+'.stat', 'r') as f:
        stat = json.load(f)
    return stat


# load dev to DataFrame

# In[8]:


def get_df_dev(path):
    dev_sub = 'dev/action_instruction.txt'
    df_dev = pd.read_csv(os.path.join(path, dev_sub), sep='\t', header=None)
    df_dev.columns=['traj', 'uttr']
    return df_dev


# main

# In[9]:


data_path_prefix = './data/fixed_order/forward_marker'
df_dev = get_df_dev(data_path_prefix)

traj_avg_len = get_length_column(df_dev, 'traj').mean()
uttr_avg_len = get_length_column(df_dev, 'uttr').mean()


# In[10]:


print(df_dev.columns)
print('traj_avg_len: %.2f'%traj_avg_len)
print('uttr_avg_len: %.2f'%uttr_avg_len)


# In[11]:


def transform_csv(csv):
    dump = pd.read_csv(csv, sep='\t')
    dump['src'] = dump['src'].apply(src_cut)
    dump.loc[:, 'out_len'] = dump['out'].apply(lambda x: len(x.split()))
    t2u = dump.loc[dump['direction']=='t2u'].rename(columns={'src':'traj'})
    u2t = dump.loc[dump['direction']=='u2t'].rename(columns={'src':'uttr'})
    print('t2u.columns: %s'%t2u.columns)
    print('u2t.columns: %s'%u2t.columns)
    return t2u, u2t


# In[29]:


def csv_stat(f_csv):
    dict_stat = dict()
    t2u, u2t = transform_csv(f_csv)
    
    vocab_cat = {'command': ['left','right','up','down'],
            'marker': ['M1', 'M2', 'M3', 'M4', 'M5'],
            'steps': ['1', '2', '3']}
    markers = ('M1', 'M2', 'M3', 'M4', 'M5')
    forward_order = [['M1'], ['M1','M2'], ['M1','M2','M3'], ['M1','M2','M3','M4'], ['M1','M2','M3','M4','M5']]
    # further explore --should be 22
    # 22 = c50(1) + c51(5) + c52(10) + c53(10) + c54(5) + c55(1)
    forward_potential_order = [['M2'], ['M2','M3'], ['M2','M3','M4'], ['M2','M3','M4','M5'],
                               ['M3'], ['M3','M4'], ['M3','M4','M5'],
                               ['M4'], ['M4','M5'], ['M5']]

    
    # average length
    dict_stat['out_uttr_len'] = t2u['out_len'].mean()
    dict_stat['out_traj_len'] = u2t['out_len'].mean()
        
    # Conjunction 
    t2u_join, u2t_join = conjunct(df_dev, t2u, u2t)
    
    flags = []
    for index, row in t2u_join.iterrows():
        flags.append(check_seq(row, markers, forward_order, forward_potential_order, vocab_cat))
        
#     u2t_accuracy = u2t_acc(u2t_join)
#     t2u_accuracy = t2u_acc(t2u_join)
    
    return dict_stat, t2u_join, u2t_join, flags


# In[33]:


# ***** ***** ***** ***** test one file **** ***** ***** *****
# csv_dir = './experiments_copy/models'
# csv_files = get_csv_file(csv_dir)
# csv = csv_files[0]
# dict_stat, t2u_join, u2t_join, flags = csv_stat(csv)


# In[70]:


# last run
def sum_csv(csv_dir):
    
    csv_files = get_csv_file(csv_dir)
    pid = os.path.split(csv_dir)[-1]
    
    logs = ''
    for csv in csv_files:
        dict_stat, t2u_join, u2t_join, flags = csv_stat(csv)
        # explore later
        log = dict_stat
        itr = get_iter(csv)
        logs = logs + itr + '\n' + str(log)+ '\n' + '\n'
        
    stat_path = os.path.join(csv_dir, f'{pid}.sum')
    with open(stat_path, 'w') as f:
        f.write(logs)


# In[71]:


# last run
parser = argparse.ArgumentParser()
parser.add_argument('--csv_dir', default='./experiments_copy/models', type=str)
args = parser.parse_args()
csv_dir = args.csv_dir

# csv_dir = './experiments_copy/models'
sum_csv(csv_dir)


# In[ ]:




