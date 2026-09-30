"""Shared cross-retailer forecasters; compact cross-modal Transformer component.

This module is not full MARIOD: no invented inventory/RL/sensor environment.
"""
import math
import torch
from torch import nn

class PooledLSTM(nn.Module):
    def __init__(self,nentities):
        super().__init__();self.embedding=nn.Embedding(nentities,8)
        self.encoder=nn.LSTM(12,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(64+8+64,64),nn.ReLU(),nn.Linear(64,16))
    def forward(self,x,entity,calendar):
        h,_=self.encoder(x)
        return self.head(torch.cat([h[:,-1],self.embedding(entity),calendar.flatten(1)],1))

class CrossModalTransformer(nn.Module):
    """Pre-LN temporal attention + modality-specific attention + horizon queries.

    Inspired by paper equations 9–12. Compact dimensions and deterministic point
    head are explicit adaptations; not the paper's 512D/six-layer full model.
    """
    def __init__(self,nentities):
        super().__init__();dim=64
        self.embedding=nn.Embedding(nentities,dim);self.domain=nn.Linear(3,dim)
        self.groups=[[0,1,2],[3,8],[4,5,6,7]]
        self.projections=nn.ModuleList([nn.Linear(len(g),dim) for g in self.groups])
        self.encoders=nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,128,.1,batch_first=True,norm_first=True),1,enable_nested_tensor=False) for _ in self.groups])
        self.cross=nn.ModuleList([nn.MultiheadAttention(dim,4,dropout=.1,batch_first=True) for _ in self.groups])
        self.query=nn.Linear(4,dim);self.gate=nn.Linear(dim,3)
        self.norm=nn.LayerNorm(dim);self.ff=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,dim))
        self.output=nn.Linear(dim,1)
        p=torch.arange(28)[:,None];freq=torch.exp(torch.arange(0,dim,2)*(-math.log(10000)/dim));pos=torch.zeros(28,dim);pos[:,0::2]=torch.sin(p*freq);pos[:,1::2]=torch.cos(p*freq);self.register_buffer('position',pos)
    def forward(self,x,entity,calendar):
        context=self.embedding(entity)+self.domain(x[:,-1,9:12])
        q=self.query(calendar)+context[:,None,:]
        streams=[]
        for cols,projection,encoder,attn in zip(self.groups,self.projections,self.encoders,self.cross):
            memory=encoder(projection(x[:,:,cols])+self.position)
            attended,_=attn(q,memory,memory,need_weights=False);streams.append(attended)
        weights=torch.softmax(self.gate(context),-1)
        fused=q+sum(weights[:,i,None,None]*v for i,v in enumerate(streams))
        fused=fused+self.ff(self.norm(fused))
        return self.output(fused).squeeze(-1)
