# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

import numpy as np
from .util import repeat_explode


class T2S(nn.Module):
    def __init__(self, A1, A2, minimal_selection, explosion_2nd):
        super(T2S, self).__init__()
        self.minimal_selection = minimal_selection
        self.explosion_2nd = explosion_2nd
        self.A1 = A1
        self.A2 = A2

    def get_lengths(self, sequence, eos_id):
        eos = sequence.eq(eos_id)
        # eos contains ones on positions where <eos> occur in the outputs, and zeros otherwise
        # eos.cumsum(dim=1) would contain non-zeros on all positions after <eos> occurred
        # eos.cumsum(dim=1) > 0 would contain ones on all positions after <eos> occurred
        # (eos.cumsum(dim=1) > 0).sum(dim=1) equates to the number of timestamps that happened after <eos> occured (including it)
        # eos.size(1) - (eos.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before eos took place
        lengths = eos.size(1) - (eos.cumsum(dim=1) > 0).sum(dim=1)

        return lengths
    
    def two_pass_explode(self, input_variable, input_lengths, n_times=3):
        # 2nd explosion
        input_variable, input_lengths, src_id = repeat_explode(input_variable, input_lengths, n_times)
        return input_variable, input_lengths, src_id
    
    def shorter_selection(self, de_out, de_other, orig_batch_size, n_times):
        
        key_len = self.A1.decoder.KEY_LENGTH
        de_out_lengths = np.array(de_other[key_len])
        key_sym = self.A1.decoder.KEY_SEQUENCE
        de_out_symbols = de_other[key_sym]
        key_attn = self.A1.decoder.KEY_ATTN_SCORE
        de_out_attn = de_other[key_attn]

        new_de_out_lengths = []
        new_de_other = dict()
        selected_overall_index = []

        # ******************** short selection ********************
        for i in range(orig_batch_size):
            batch_lengths = de_out_lengths[i*n_times : (i+1) * n_times]

            ind = np.argmin(batch_lengths)
            overall_ind = i*n_times + ind
            selected_overall_index.append(overall_ind)
            new_de_out_lengths.append(de_out_lengths[overall_ind])

        # reshape
        new_de_other[key_sym] = []
        new_de_other[key_attn] = []
        new_de_other[key_len] = new_de_out_lengths
        new_de_out = []

        indices = torch.tensor(selected_overall_index).to(de_out[0].device)
        for i in range(len(de_out)):
            select = torch.index_select(de_out[i], 0, indices)
            new_de_out.append(select)

        for i in range(len(de_out_symbols)):
            select = torch.index_select(de_out_symbols[i], 0, indices)
            new_de_other[key_sym].append(select)

        for i in range(len(de_out_attn)):
            select = torch.index_select(de_out_attn[i], 0, indices)
            new_de_other[key_attn].append(select)

        return new_de_out, new_de_other
        

    def forward(self, input_variable, input_lengths, target_variable, teacher_forcing_ratio=0.0, presorted=False):
        # turn off sampling in the teacher or in the student
        # when needed.
        A1 = self.A1
        
        # original batch_size
        orig_batch_size = input_variable.size(0)

        # two_pass_explode
        if self.minimal_selection:
            input_variable, input_lengths, src_id = self.two_pass_explode(input_variable, input_lengths, self.explosion_2nd)

        with torch.no_grad():
            teacher_decoder_outputs, _, teacher_other = A1(
                input_variable, input_lengths, None, 0.0, presorted=presorted)

        # shorter_utterance_selection
        if self.minimal_selection:
            teacher_decoder_outputs, teacher_other = self.shorter_selection(teacher_decoder_outputs, teacher_other, orig_batch_size, self.explosion_2nd)
            
        sequence_tensor = torch.stack(teacher_other['sequence']).squeeze(2).permute(1, 0)

        t_out_lengths = self.get_lengths(
            sequence_tensor, A1.decoder.eos_id)

        # NOTE: we increase len by 1 so that the final <eos> is also
        # fed into the student. At the same time, it might be the case that
        #  the teacher never produced <eos>. In tat case, we cap length.
        max_len = len(teacher_other['sequence'])
        t_out_lengths.add_(1).clamp_(max=max_len)

        student_decoder_outputs, _, student_other = self.A2(sequence_tensor, t_out_lengths, target_variable=target_variable,
                                                           teacher_forcing_ratio=teacher_forcing_ratio)
        student_other['teacher_decoder'] = teacher_other['sequence']
        student_other['teacher_decoder_outputs'] = teacher_decoder_outputs
        student_other['teacher_dict'] = teacher_other
        student_other['teacher_lengths'] = t_out_lengths

        return student_decoder_outputs, None, student_other
