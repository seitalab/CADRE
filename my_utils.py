import os
import random
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  """ Encoder module with/without self-attention function to encode omic information.
      Expencoder -> Encoder 
  """

  def __init__(
      self, omc_size, ptw_ids, hidden_dim=200, dropout_rate=0.5, embedding_dim=200,
      use_attention=True, attention_size=128, attention_head=8, init_gene_emb=True,
      use_cntx_attn=True,  use_hid_lyr=False, use_relu=False,
      repository='gdsc'):
        
        super().__init__()
    
        gene_emb_pretrain = np.genfromtxt('data/input/exp_emb_gdsc.csv', delimiter=',')
     

        self.layer_emb = nn.Embedding.from_pretrained(torch.FloatTensor(gene_emb_pretrain), freeze=True, padding_idx=0)

        self.layer_dropout_0 = nn.Dropout(p=dropout_rate)
     
        self.layer_w_0 = nn.Linear(in_features=embedding_dim,out_features=attention_size,bias=True)

        self.layer_beta = nn.Linear(in_features=attention_size,out_features=attention_head,bias=True)
      
        self.layer_emb_ptw = nn.Embedding(num_embeddings=max(ptw_ids)+1,embedding_dim=attention_size)


  def forward(self, omc_idx, ptw_ids):
    """
    Parameters
    ----------
    omc_idx: int array with shape (batch_size, num_omc)
      indices of perturbed genes in the omic data of samples

    Returns
    -------

    """

    E_t = self.layer_emb(omc_idx) #(batch_size, num_omc, embedding_dim)
    
    E_t = torch.unsqueeze(E_t,1) #(batch_size, 1, num_omc, embedding_dim)
    
    E_t = E_t.repeat(1,ptw_ids.shape[1],1,1) #(batch_size, num_drg, num_omc, embedding_dim)
    


    Ep_t = self.layer_emb_ptw(ptw_ids) #(1, num_drg, attention_size)
    
    Ep_t = torch.unsqueeze(Ep_t,2) #(1, num_drg, 1, attention_size)
    
    Ep_t = Ep_t.repeat(omc_idx.shape[0],1,omc_idx.shape[1],1) #(batch_size, num_drg, num_omc, attention_size)
    
    E_t_1 = torch.tanh( self.layer_w_0(E_t) + Ep_t) #(batch_size, num_drg, num_omc, attention_size)



    A_omc = self.layer_beta(E_t_1) #(batch_size, num_drg, num_omc, attention_head)

    A_omc = F.softmax(A_omc, dim=2) #(batch_size, num_drg, num_omc, attention_head)
    A_omc = torch.sum(A_omc, dim=3, keepdim=True) #(batch_size, num_drg, num_omc, 1)

    #(batch_size, num_drg, 1, num_omc) * (batch_size, num_drg, num_omc, embedding_dim)
    #=(batch_size, num_drg, 1, embedding_dim)

    self.Amtr = torch.squeeze(A_omc, 3) #(batch_size, num_drg, num_omc)

    emb_omc = torch.sum(torch.matmul(A_omc.permute(0,1,3,2), E_t), dim=2, keepdim=False) #(batch_size, num_drg, embedding_dim)

    hid_omc = self.layer_dropout_0(emb_omc)

    return hid_omc


class Decoder(nn.Module):
  """ Encoder module to decode the drug response from the concatenation of genome hidden layer state.

  """

  def __init__(self, hidden_dim, drg_size):
    """
    Parameters
    ----------
    hidden_dim: input hidden layer dimension of single omic type.
    drg_size: number of output drugs to be predicted.
    dropout_rate: dropout rate of the intermediate layer.

    """

    super().__init__()

    self.layer_emb_drg = nn.Embedding(
        num_embeddings=drg_size,
        embedding_dim=hidden_dim)

    self.drg_bias = nn.Parameter(torch.zeros(drg_size)) #(num_drg)


  def forward(self, hid_omc, drg_ids):
    """
    """
    #hid_omc: (batch_size, num_drg, hidden_dim_enc)

    E_t = self.layer_emb_drg(drg_ids) # (1, num_drg, hidden_dim_enc)

    E_t = E_t.repeat(hid_omc.shape[0],1,1) # (batch_size, num_drg, hidden_dim_enc)

    logit_drg = torch.matmul(
        hid_omc.view(hid_omc.shape[0], hid_omc.shape[1], 1, hid_omc.shape[2]),
        E_t.view(E_t.shape[0], E_t.shape[1], E_t.shape[2], 1))

    logit_drg = torch.sum(logit_drg, dim=2, keepdim=False) # (batch_size, num_drg)
    logit_drg = torch.sum(logit_drg, dim=2, keepdim=False)

    drg_bias = torch.unsqueeze(self.drg_bias,0) # (1, num_drg)
    drg_bias = drg_bias.repeat(hid_omc.shape[0],1) #(batch_size, num_drg)

    return logit_drg

