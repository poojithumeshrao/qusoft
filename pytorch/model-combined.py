import torch
import torch.nn as nn
from torch.nn.functional import normalize
import pennylane as qml
import sys
import torch.distributed as dist

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_default_tensor_type(torch.DoubleTensor)
#size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

#dist.init_process_group("gloo", rank=rank, world_size=size)

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.rand(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.rand(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E 
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        return x


class QuantumAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wires = 12
        self.dev = qml.device('default.qubit', wires=self.wires)
        self.n_features = 2**(self.wires//3)
        self.shape =  qml.StronglyEntanglingLayers.shape(n_wires=len(range(2*(self.wires//3))),n_layers = 1)
        self.weights = nn.Parameter(torch.empty(size=self.shape))
        self.qnode = qml.QNode(self.qnn_attention,self.dev,interface='torch')

    def feature_map(self,f1 = None,f2 = None,q1 = None,q2 = None):
        #print(f1.shape)
        qml.MottonenStatePreparation(state_vector=f1, wires=q1)
        qml.MottonenStatePreparation(state_vector=f2, wires=q2)

    def measure(self,q1,q2,q3):
        for i in range(len(q1)):
            qml.CNOT(wires=[q1[i],q2[i]])
            qml.Hadamard(wires=[q3[i]])

        for i in range(len(q1)):
            qml.Toffoli(wires=[q2[i],q3[i],q1[i]])

        for i in range(len(q1)):
            qml.CNOT(wires=[q1[i],q2[i]])
            qml.Hadamard(wires=[q3[i]])

    def qnn_attention(self,f1,f2,w,wires):
        self.feature_map(f1,f2,range(wires//3),range(wires//3,2*(wires//3)))
        qml.StronglyEntanglingLayers(weights=w, wires=range(2*(wires//3)), imprimitive=qml.ops.CZ)
        self.measure(range(wires//3),range(wires//3,2*(wires//3)),range(2*(wires//3),wires))
        return qml.probs(wires=range(2*(wires//3),wires))

    def forward(self,x):
        o = torch.empty((0))
        for sample in x:
            f1,f2 = sample[:self.n_features],sample[self.n_features:]
            o = torch.cat((o,torch.Tensor(self.qnode(f1=f1,f2=f2,w=self.weights,wires=self.wires)[0]).reshape(1)))
            #outputs = q.get()
            #outputs = torch.cat((outputs,o))
            #q.put(outputs)
            
        return o

class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True).double()
        self.qnn = QuantumAttention(args)
        self.qnn.share_memory()
        
    def forward(self, x):
        m, s, e = x.shape
        t = torch.empty((0,s,2*e),dtype=torch.double)
        x = x.double()

        outputs = torch.empty((0),dtype=torch.double)

        #print(x.shape)

        processes = []
        #q = mp.Queue()
        print(x.shape)
        for bat in x:
            for vec in normalize(bat):
                #q.put(outputs)
                #with mp.Pool(64) as p:
                #    o = p.map(self.qnn,torch.stack((vec.repeat(bat.shape[0],1),normalize(bat)),1).reshape(s,2*e))
                outputs = torch.cat((outputs,self.qnn(torch.stack((vec.repeat(bat.shape[0],1),normalize(bat)),1).reshape(s,2*e))))
            #print(outputs.shape)
        #for op in o:
        #    outputs = torch.cat((outputs,op))
        
        #t = torch.cat((t,torch.stack((vec.repeat(bat.shape[0],1),normalize(bat)),1).reshape(1,s,2*e)))

        
        # xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        # xq = xq.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
        # xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, K, E -> B, K, H, HE
        # xk = xk.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE

        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, V, E -> B, V, H, HE
        xv = xv.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE

        # xq = xq.reshape([-1, s, self.head_embed_dim])  # B, H, Q, HE -> (BH), Q, HE
        # xk = xk.reshape([-1, s, self.head_embed_dim])  # B, H, K, HE -> (BH), K, HE
        xv = xv.reshape([-1, s, self.head_embed_dim])  # B, H, V, HE -> (BH), V, HE

        
        # xk = xk.transpose(1, 2)  # (BH), K, HE -> (BH), HE, K

        #x_attention = self.qnn(t.reshape([-1,2*e]))
        #del t
        x_attention = outputs.reshape(m,s,s)
        #x_attention = xq.bmm(xk)  # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K
        x_attention = torch.softmax(x_attention, dim=-1)
        
        x = x_attention.bmm(xv)  # (BH), Q, K . (BH), V, HE -> (BH), Q, HE
        x = x.reshape([-1, self.n_attention_heads, s, self.head_embed_dim])  # (BH), Q, HE -> B, H, Q, HE
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(m, s, e)  # B, Q, H, HE -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfAttention(args)
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # Skip connections
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))  # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)], nn.LayerNorm(args.embed_dim))
        self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(args)

    def forward(self, x):
        
        x = self.embedding(x)
        #print(x[0])
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x
