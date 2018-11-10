import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_





# construct neuron network

def scaled_dot_attention(Q, K, V, mask):
    assert Q.size()[-1] == K.size()[-1]
    assert len(Q.size()) == 3 and len(K.size()) == 3 and len(V.size()) == 3
    dk = torch.tensor(K.size()[-1], dtype=torch.float32, requires_grad=False).cuda()
    out = torch.matmul(Q,K.permute(0,2,1)) / torch.sqrt(dk) 
    if mask is not None:
        out.masked_fill_(mask, -float('inf'))
    return torch.matmul(F.softmax(out, dim=-1), V)

def positional_encoding(d_model, pos):
    assert d_model % 2 == 0
    pos = torch.tensor(pos, dtype=torch.float32, requires_grad=False)
    pe = torch.zeros([1,d_model], dtype=torch.float32, requires_grad=False)
    for i in range(D_MODEL//2):
        a = torch.tensor(10000, dtype=torch.float32, requires_grad=False)
        b = torch.tensor(2.*i/float(D_MODEL), dtype=torch.float32, requires_grad=False)
        c = pos / torch.pow(a, b)
        pe[0, 2*i] = torch.sin(c)
        pe[0, 2*i+1] = torch.cos(c)
    return pe
                            
class Transformer(nn.Module):

    def __init__(self, layer_num, dk, dv, dm, h, p_drop, d_ff, use_mask, use_cuda=True, posi_cache_length=200):
        super(Transformer, self).__init__()
#         for construct cache positional encoding matrix.
        self.d_model = dm
        self.use_cuda = use_cuda
        
        self.encoder = Stack_Encoder(layer_num, dk, dv, dm, h, p_drop, d_ff)
        self.decoder = Stack_Decoder(layer_num, dk, dv, dm, h, p_drop, d_ff, use_mask)
        self.emb_drop = nn.Dropout(p_drop)
        self.init_pos_mat(posi_cache_length)

    def forward(self, Q, K):
    
    #         encoder
        batch, K_len, d = K.size()
#         pos matrix will fit the batch size
#         without pos.repeat() is faster
        try:
            K = K + self.get_pos_mat(K_len)
        except RuntimeError, e:
            if e.message == 'TensorIterator expected type torch.cuda.FloatTensor but got torch.FloatTensor':
                if K.is_cuda != self.get_pos_mat(K_len).is_cuda:
                    print('Make sure cache positional matrix is same type of tensor with input, both cuda tensor or not.\nBy setting argument use_cuda=True to set cache positional encoding matrix as a cuda tensor.')
            raise
        K = self.emb_drop(K)
        
        en_out = self.encoder(K)
        
#         decoder
        batch, Q_len, d = Q.size()
        
        try:
            Q = Q + self.get_pos_mat(Q_len)
        except RuntimeError, e:
            if e.message == 'TensorIterator expected type torch.cuda.FloatTensor but got torch.FloatTensor':
                if Q.is_cuda != self.get_pos_mat(K_len).is_cuda:
                    print('Make sure cache positional matrix is same type of tensor with input, both cuda tensor or not.\nBy setting argument use_cuda=True to set cache positional encoding matrix as a cuda tensor.')
            raise
        
        Q = self.emb_drop(Q)
        
        de_out = self.decoder(Q, en_out)
        return de_out
    
#     To speed up the positional encoding by construct an cache matrix. 
    def init_pos_mat(self, cache_length):
        print('init postional matrix with length : %d ' % cache_length)
        self.positional_matrix = torch.cat([positional_encoding(self.d_model, i) for i in range(0,cache_length)], dim=0)
        self.positional_matrix.requires_grad = False
        if self.use_cuda:
            self.positional_matrix = self.positional_matrix.cuda()
            
        
    def get_pos_mat(self, length):
        if length > self.positional_matrix.shape[0]:
            print('input sequence length reach positional matrix maximum length. %d ' % length)
            ret = torch.cat([positional_encoding(self.d_model, i) for i in range(length)], dim=0)
            ret.requires_grad = False
            print('Increase positional matrix maximum length. %d ' % length)
            self.positional_matrix = ret
            if self.use_cuda:
                self.positional_matrix = self.positional_matrix.cuda()
            return ret
        else:
            return self.positional_matrix[:length]
        

    
    

class Stack_Encoder(nn.Module):
    """
    Stacked Encoder
    """
    def __init__(self, layer_num, dk, dv, dm, h, p_drop, d_ff):
        super(Stack_Encoder, self).__init__()
        self.encoders = nn.ModuleList([Encoder(dk, dv, dm, h, p_drop, d_ff) for i in range(layer_num)])

    def forward(self, K):
        # ModuleList can act as an iterable, or be indexed using ints
        for lay in self.encoders:
            K = lay(K)
        return K         
    
class Encoder(nn.Module):
    def __init__(self, dk, dv, dm, h, p_drop, d_ff):
        super(Encoder, self).__init__()
#         attention residual block
        self.multi_head_attention_layer = Multi_Head_attention_layer(dk, dv, dm, h)
        self.attention_norm_lay = nn.LayerNorm([dm,])
        self.att_drop = nn.Dropout(p_drop)
#         feed forward residual block
        self.fcn = PositionwiseFeedForward(dm, d_ff)
        self.linear_drop = nn.Dropout(p_drop)
        self.ff_norm_lay = nn.LayerNorm([dm, ])
        

    def forward(self, K):
#         attention
        attention_out = self.multi_head_attention_layer(K, K, K, mask=None)
        attention_out = self.att_drop(attention_out)
        att_out = self.attention_norm_lay(K + attention_out)
#         feed forward
        linear_out = self.fcn(att_out)
        linear_out = self.linear_drop(linear_out)
        out = self.ff_norm_lay(att_out + linear_out)
        out = att_out + linear_out
    
        return out
class Stack_Decoder(nn.Module):
    """
    Stacked Decoder
    """
    def __init__(self, layer_num, dk, dv, dm, h, p_drop, d_ff, use_mask):
        super(Stack_Decoder, self).__init__()
        self.decoders = nn.ModuleList([Decoder(dk, dv, dm, h, p_drop, d_ff, use_mask) for i in range(layer_num)])
        
        
    def forward(self, Q, encoder_out):
        # ModuleList can act as an iterable, or be indexed using ints
        for lay in self.decoders:
            Q = lay(Q, encoder_out)
        return Q           

class Decoder(nn.Module):
    def __init__(self, dk, dv, dm, h, p_drop, d_ff, use_mask):
        super(Decoder, self).__init__()
        self.use_mask = use_mask
        
#         query attention residual block
        self.Q_attention_lay = Multi_Head_attention_layer(dk, dv, dm, h)
        self.Q_attention_norm_lay = nn.LayerNorm([dm, ])
        self.Q_att_drop = nn.Dropout(p_drop)
    
#         query key attention residual block
        self.QK_attention_lay = Multi_Head_attention_layer(dk, dv, dm, h)
        self.QK_attention_norm_lay = nn.LayerNorm([dm, ])
        self.QK_att_drop = nn.Dropout(p_drop)
        
    
#         feed forward residual block
        self.fcn = PositionwiseFeedForward(dm, d_ff)
        self.ff_norm_lay = nn.LayerNorm([dm, ])
        self.linear_drop = nn.Dropout(p_drop)
        

    def forward(self, Q, encoder_out):
        if self.use_mask:
            batch, Q_len, d = Q.size()
            mask = self.mask_matrix(batch, Q_len)
        else:
            mask = None
#         query attention
        Q_attention_out = self.Q_attention_lay(Q, Q, Q, mask=mask)
        Q_attention_out = self.Q_att_drop(Q_attention_out)
        Q_att_out = self.Q_attention_norm_lay(Q + Q_attention_out)
#         query key attention
        QK_attention_out = self.QK_attention_lay(Q_att_out, encoder_out, encoder_out, mask=None)
        QK_attention_out = self.QK_att_drop(QK_attention_out)
        QK_att_out = self.QK_attention_norm_lay(Q_att_out + QK_attention_out)
        
#         feed forward
        linear_out = self.fcn(QK_att_out)
        out = self.ff_norm_lay(QK_att_out + linear_out)
        return out
    def mask_matrix(self, batch, Q_len):
#         ByteTensor
        mask = torch.zeros([1, Q_len, Q_len], dtype=torch.uint8, requires_grad=False)
        for i in range(Q_len):
            mask[0,i,i+1:] = 1
        return mask.repeat(batch,1, 1).cuda()


class Multi_Head_attention_layer(nn.Module):
    def __init__(self, dk, dv, dm, h):
        super(Multi_Head_attention_layer, self).__init__()
        self.Q_linears = nn.ModuleList([nn.Linear(dm, dk) for i in range(h)])
        self.K_linears = nn.ModuleList([nn.Linear(dm, dk) for i in range(h)])
        self.V_linears = nn.ModuleList([nn.Linear(dm, dv) for i in range(h)])
        self.output_linear = nn.Linear(h*dv, dm)
                            

    def forward(self, Q_input, K_input, V_input, mask):
        buf = []
        for Q_linear, K_linear, V_linear in zip(self.Q_linears, self.K_linears, self.V_linears):
            Q = Q_linear(Q_input)
            K = K_linear(K_input)
            V = V_linear(V_input)
            buf.append(scaled_dot_attention(Q, K, V, mask))
        
        buf = torch.cat(buf,dim=-1)
        out = self.output_linear(buf)
        
        return out      
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.cnn1 = nn.Conv1d(d_model, d_ff, 1)
        self.cnn2 = nn.Conv1d(d_ff, d_model, 1)
                            

    def forward(self, x):
        bat,seq_len,d = x.size()
        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = x.permute(0,2,1)
        
        return x      
    

    
    
# Transformer paper baseline hyper-parameters
# STACKED_NUM = 6
# H = 8
# D_MODEL = 512
# DK = DV = D_MODEL//H
# P_DROP = 0.1
# D_FF = 2048




    
# bat = 3
# STACKED_NUM = 2
# DK=DV=D_MODEL
# P_DROP=0.1
# D_FF = D_MODEL*4
# H=8
# Q = torch.rand([bat, 13, D_MODEL]).cuda()
# K = torch.rand([bat, 19, D_MODEL]).cuda()
# net = Transformer(STACKED_NUM, DK, DV, D_MODEL, H, P_DROP, D_FF, use_mask=True, use_cuda=True).cuda()
# o = net(Q, K)
# print(o.size())

# Q = torch.rand([bat, 47, D_MODEL]).cuda()
# K = torch.rand([bat, 88, D_MODEL]).cuda()
# o = net(Q, K)
# print(o.size())
# # # print o
# # def count_parameters(model):
# #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print(count_parameters(net))

