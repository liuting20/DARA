# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
from transformers.models.bert.configuration_bert import BertConfig
from models.language_model.med import BertModel

from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import os

class BERT(nn.Module):
    def __init__(self, args,name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num, med_config = 'configs/med_config.json',config = None,text_adapters = None):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        self.config = config
        med_config = BertConfig.from_json_file(med_config)
        self.bert = BertModel.from_pretrained(med_config.bert_path,config=med_config, add_pooling_layer=False, adapter_config=config, adapters=text_adapters,args = args)


    def forward(self,text_data_tensors, text_data_mask):

        if self.enc_num > 0:
            xs = self.bert(text_data_tensors, token_type_ids=None, attention_mask=text_data_mask, return_dict = True)
        else:
            xs = self.bert.embeddings.word_embeddings(text_data_tensors)

        mask = text_data_mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_bert(args,config,text_adapters = None):
    train_bert = args.lr_bert > 0
    bert = BERT(args,args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num,config = config,text_adapters = text_adapters)
   
    return bert



