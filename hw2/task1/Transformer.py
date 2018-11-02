import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from constants import D_MODEL,PE_MAT_CACHE


def positional_encoding(pos):
    assert D_MODEL % 2 == 0
    pos = torch.tensor(pos, dtype=torch.float32, requires_grad=False)
    pe = torch.zeros([1,D_MODEL], dtype=torch.float32, requires_grad=False)
    for i in range(D_MODEL//2):
        a = torch.tensor(10000, dtype=torch.float32, requires_grad=False)
        b = torch.tensor(2.*i/float(D_MODEL), dtype=torch.float32, requires_grad=False)
        c = pos / torch.pow(a, b)
        pe[0, 2*i] = torch.sin(c)
        pe[0, 2*i+1] = torch.cos(c)
    return pe
def get_pos_mat(length):
    if length > PE_MAT_CACHE:
        print 'sequence length reach PE_MAT_CACHE. %d ' % length
        ret = torch.cat([positional_encoding(i) for i in range(length)], dim=0).cuda()
        ret.requires_grad = False
        global PE_CACHE_MATRIX
        PE_CACHE_MATRIX = ret
        return ret
    else:
        return PE_CACHE_MATRIX[:length]
def mask_matrix(Q_mask_len, batch, Q_len, K_len):
    assert len(Q_mask_len.shape) == 1
#     ByteTensor
    mask = torch.zeros([batch, Q_len, K_len], dtype=torch.uint8, requires_grad=False)
    for i in range(batch):
        seq_len = Q_mask_len[i]
        if seq_len == Q_len:
            continue
        mask[i,seq_len:,:] = 1
        
    return mask.cuda()

    
PE_CACHE_MATRIX = torch.cat([positional_encoding(i) for i in range(0,PE_MAT_CACHE)], dim=0).cuda()
PE_CACHE_MATRIX.requires_grad = False

# construct neuron network

def scaled_dot_attention(Q, K, V, mask=None):
    assert Q.size()[-1] == K.size()[-1]
    assert len(Q.size()) == 3 and len(K.size()) == 3 and len(V.size()) == 3
    dk = torch.tensor(K.size()[-1], dtype=torch.float32, requires_grad=False).cuda()
    out = torch.matmul(Q,K.permute(0,2,1)) / torch.sqrt(dk) 
    if mask is not None:
        out.masked_fill_(mask, -float('inf'))
#         print 'mask out', out.shape, V.shape, F.softmax(out, dim=-1).shape, torch.matmul(F.softmax(out, dim=-1), V).shape
#         print F.softmax(out, dim=-1)
        
    return torch.matmul(F.softmax(out, dim=1), V)
                            
class Transformer(nn.Module):

    def __init__(self, layer_num, dk, dv, dm, h, p_drop, d_ff):
        super(Transformer, self).__init__()
        
        self.encoder = Stack_Encoder(layer_num, dk, dv, dm, h, p_drop, d_ff)
        self.decoder = Stack_Decoder(layer_num, dk, dv, dm, h, p_drop, d_ff)
        self.emb_drop = nn.Dropout(p_drop)

    def forward(self, Q, K, Q_mask_len):
    
    #         encoder
        batch, K_len, d = K.size()
#         pos matrix will fit the batch size
#         without pos.repeat() is faster
        K = K + get_pos_mat(K_len)
        K = self.emb_drop(K)
        
        en_out = self.encoder(K)
        
#         decoder
        batch, Q_len, d = Q.size()
        
        Q = Q + get_pos_mat(Q_len)
        Q = self.emb_drop(Q)
        
        mask = mask_matrix(Q_mask_len, batch, Q_len, K_len)
        de_out = self.decoder(Q, en_out, mask)
        return de_out
         
        


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
        attention_out = self.multi_head_attention_layer(K, K, K)
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
    Stacked Encoder
    """
    def __init__(self, layer_num, dk, dv, dm, h, p_drop, d_ff):
        super(Stack_Decoder, self).__init__()
        self.decoders = nn.ModuleList([Decoder(dk, dv, dm, h, p_drop, d_ff) for i in range(layer_num)])
        
        
    def forward(self, Q, encoder_out, mask):
        # ModuleList can act as an iterable, or be indexed using ints
        for lay in self.decoders:
            Q = lay(Q, encoder_out, mask=mask)
        return Q           

class Decoder(nn.Module):
    def __init__(self, dk, dv, dm, h, p_drop, d_ff):
        super(Decoder, self).__init__()
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
        

    def forward(self, Q, encoder_out, mask):
#         query attention
        Q_attention_out = self.Q_attention_lay(Q, Q, Q, mask=None)
        Q_attention_out = self.Q_att_drop(Q_attention_out)
        Q_att_out = self.Q_attention_norm_lay(Q + Q_attention_out)
#         query key attention
        QK_attention_out = self.QK_attention_lay(Q_att_out, encoder_out, encoder_out, mask=mask)
        QK_attention_out = self.QK_att_drop(QK_attention_out)
        QK_att_out = self.QK_attention_norm_lay(Q_att_out + QK_attention_out)
        
#         feed forward
        linear_out = self.fcn(QK_att_out)
        out = self.ff_norm_lay(QK_att_out + linear_out)
        return out

class Multi_Head_attention_layer(nn.Module):
    def __init__(self, dk, dv, dm, h):
        super(Multi_Head_attention_layer, self).__init__()
        self.Q_linears = nn.ModuleList([nn.Linear(dm, dk) for i in range(h)])
        self.K_linears = nn.ModuleList([nn.Linear(dm, dk) for i in range(h)])
        self.V_linears = nn.ModuleList([nn.Linear(dm, dv) for i in range(h)])
        self.output_linear = nn.Linear(h*dv, dm)
                            

    def forward(self, Q_input, K_input, V_input, mask=None):
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
        bat,seq_len,_ = x.size()
        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = x.permute(0,2,1)
        
        return x      
    
bat = 3
# Q = torch.rand([bat, 13, D_MODEL]).cuda()
# V = torch.rand([bat, 19, D_MODEL]).cuda()
# Q_mask_len = torch.tensor([2,3,4]).cuda()
# print Q.shape, V.shape, Q_mask_len.shape
# net = Transformer(STACKED_NUM, DK, DV, D_MODEL, H, P_DROP, D_FF).cuda()
# o = net(Q, V, Q_mask_len)
# print(o.size())
# # print o
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(count_parameters(net))

