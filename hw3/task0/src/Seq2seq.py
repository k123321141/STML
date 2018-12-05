import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from constants import VOCAB_DIM
# construct neuron network
class Seq2seq(nn.Module):

    def __init__(self, dm, num_lay):
        super(Seq2seq, self).__init__()
#         for construct cache positional encoding matrix.
        self.emb = nn.Embedding(VOCAB_DIM+1, dm, padding_idx=0)
        self.encoder = nn.GRU(dm, dm, num_lay)
        self.decoder = nn.GRU(dm, dm, num_lay)
        self.linear = nn.Linear(dm, VOCAB_DIM+1)

    def forward(self, Q, K):
        K = self.emb(K)
        Q = self.emb(Q)
        
        
        en_out, hn = self.encoder(K.permute(1,0,2)) 
        Q, _ = self.decoder(Q.permute(1,0,2), hn)
    
        
        y = self.linear(Q.permute(1,0,2))
        return y
