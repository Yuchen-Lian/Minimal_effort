#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import sys

sys.path.insert(0, './pytorch-seq2seq/')

import argparse
import pickle

import torch
import torchtext
from torchtext.data import Field

import t2s.util as util
import pandas as pd
import os


# set vocab
def set_vocab(all_dataset_vocab):
    field = Field(preprocessing=lambda x: ['<sos>'] + x + ['<eos>'], unk_token=None, batch_first=True,
                  include_lengths=True, pad_token='<pad>')
    vocab = torchtext.data.TabularDataset(
        path=all_dataset_vocab, format='tsv',
        fields=[('src', field), ('tgt', field)])
    field.build_vocab(vocab, max_size=50000)
    print("Vocab: {}".format(field.vocab.stoi), flush=True)
    return field, vocab

# set dev_dataset
def set_devdata(data_path_prefix, vocab):
    language = data_path_prefix.split('/')[-1]
    t2u_dev_path = f'{data_path_prefix}/dev/action_instruction.txt'
    dev_dataset = torchtext.data.TabularDataset(
        path=t2u_dev_path, format='tsv',
        fields=[('src', field), ('tgt', field)])
    return language, dev_dataset

# load model
def load_model(init_A1, init_A1_from_A2, device):
    p1 = init_A1 and init_A1_from_A2
    p2 = (not init_A1) and (not init_A1_from_A2)
    if p1 or p2: 
        raise ValueError("One file_dir is needed.")
    if init_A1:
        with open(init_A1, "rb") as fin:
            m = pickle.load(fin)
        if hasattr(m, "A1"):
            A1 = m.A1
            print('Loaded A1 as submodel')
        else:
            A1 = m
        A1.to(device)
    if init_A1_from_A2:
        with open(init_A1_from_A2, "rb") as fin:
            A1 = pickle.load(fin).A2.to(device)
        print('Loaded A1 as an A2 submodel')
    A1.flatten_parameters()
    return A1


def dump_agent_csv(A, iterator, output_file, field, instruction_explosion_rate=10):
    def id_to_text(ids):
        return [field.vocab.itos[x.item()] for x in ids]

    stats = util.LangStats()
    src_, out_, inf_ = [], [], []

    with torch.no_grad():
        batch_generator = iterator.__iter__()
        bat = 0
        for batch in batch_generator:
            bat = bat + 1
            tasks = [(batch.src, instruction_explosion_rate, 't2u'),
                     (batch.tgt, 1, 'u2t')]
            for ((src, length), explosion_rate, name) in tasks:
                src, length, src_id = util.repeat_explode(src, length, explosion_rate)
                _1, _2, other = A.forward(src, length, None, 0.0)
                out = torch.stack(other['sequence']).squeeze(2).permute(1, 0)
                for i in range(src.size(0)):
                    src_seq = util.cut_after_eos(id_to_text(src[i, :]))
                    src_str = ' '.join(src_seq)
                    out_seq = util.cut_after_eos(id_to_text(out[i, :]))
                    out_str = ' '.join(out_seq)

                    inf_series = pd.Series([bat, src_id[i], name], name='n%d' % i,
                                           index=['batch_id', 'src_id', 'direction'])
#                     src_series = pd.Series(src_seq, name='n%d' % i, index=['src_%d' % j for j in range(len(src_seq))])
#                     out_series = pd.Series(out_seq, name='n%d' % i, index=['out_%d' % j for j in range(len(out_seq))])
                    src_series = pd.Series(src_str, name='n%d' % i, index=['src'])
                    out_series = pd.Series(out_str, name='n%d' % i, index=['out'])

                    inf_.append(inf_series)
                    src_.append(src_series)
                    out_.append(out_series)

                    if name == "t2u":
                        stats.push_stat(out_seq)

    df_inf = pd.DataFrame(inf_)
    df_src = pd.DataFrame(src_)
    df_out = pd.DataFrame(out_)

    result = pd.concat([df_inf, df_src, df_out], axis=1)
    result.to_csv(output_file, sep='\t', na_rep='NaN')
    
    stat_dir = output_file + '.stat'
    with open(stat_dir, 'w') as f:
        f.write(stats.get_json())


def dump_csv(agent, dataset, path):
    iterator = torchtext.data.BucketIterator(dataset, batch_size=1024, sort=True, sort_within_batch=True,
                                             sort_key=lambda x: len(x.src),
                                             device=("cuda" if torch.cuda.is_available() else "cpu"), repeat=False)
    return dump_agent_csv(agent, iterator, path, field)


def get_model_file(model_dir):
    f = []
    for dir_path, subpaths, files in os.walk(model_dir):
        f.append((dir_path, files))
    dir_, files = f[0]
    
    f_models=[]
    for f in files:
        (fname, ext) = os.path.splitext(f)
        if ext == '.p' or '.iteration_' in ext:
            m = os.path.join(dir_, f)
            f_models.append(m)
    return f_models


def dump_(m, dev_dataset):
    if os.path.splitext(m)[-1] == '.p':
        A1 = load_model(m, None, device)
    else:
        A1 = load_model(None, m, device)
    dump_path = m + '.csv'
    dump_csv(A1, dev_dataset, dump_path)


parser = argparse.ArgumentParser()
parser.add_argument('--pid', default='p40', type=str)
args = parser.parse_args()

model_dir= f'./experiments_copy/{args.pid}'
f_models = get_model_file(model_dir)
if f_models is None:
    print('empty dir')
else:
    print(f_models)

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    data_path_prefix = './data/fixed_order/forward_marker'
    all_dataset_vocab = './data/fixed_order/vocabulary.txt'
    field, vocab = set_vocab(all_dataset_vocab)
    language, dev_dataset = set_devdata(data_path_prefix, vocab)

    for m in f_models:
        dump_(m, dev_dataset)
        print('finished writing %s' % m)


