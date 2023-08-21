import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import *
from qiskit.circuit.library import ZZFeatureMap,TwoLocal,EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from torch.nn.functional import normalize

torch.set_default_tensor_type(torch.DoubleTensor)

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


class SelfAttention(nn.Module):
    def encode_circ_raw(self,num_features = 4,param_name = ''):
        vec = ParameterVector(param_name,num_features)
        qc = RawFeatureVector((num_features))
        qc.assign_parameters(vec,inplace=True)
        #print(qc.parameters)
        return qc

    def encode_circ_zz(self,num_features = 2):
        qc = EfficientSU2(num_features*2,parameter_prefix='input',reps=5)
        return qc

    def entangle_circ(self,num_features = 2):
        qc = TwoLocal(num_features * 2,reps = 1,rotation_blocks = 'ry',entanglement_blocks = 'cx',skip_final_rotation_layer = True)
        return qc

    def measure_circ(self,reg_a,reg_b,reg_c, num_features = 2):
    
        mc = QuantumCircuit(reg_a,reg_b,reg_c)

        #Perform controlled SWAP test for measuring entanglement
        
        for i in range(num_features):
            mc.cx( reg_a[i],reg_b[i])
            mc.h(reg_c[i])

        for i in range(num_features):
            mc.ccx(reg_b[i],reg_c[i],reg_a[i])

        for i in range(num_features):
            mc.cx(reg_a[i],reg_b[i])
            mc.h(reg_c[i])
        return mc

    def get_QNN(self,num_features=4):
        
        reg_a = QuantumRegister(num_features)
        reg_b = QuantumRegister(num_features)
        reg_c = QuantumRegister(num_features)
        
        qc = QuantumCircuit(reg_a,reg_b,reg_c)
        #fm = self.encode_circ_zz(num_features=num_features)
        f1 = self.encode_circ_raw(2**num_features,param_name='reg_a')
        f2 = self.encode_circ_raw(2**num_features,param_name='reg_b')

        qc.compose(f1,inplace=True,qubits=reg_a)
        qc.compose(f2,inplace=True,qubits=reg_b)
        
        pqc = self.entangle_circ(num_features=num_features)
        qc.compose(pqc,inplace=True)
        qc.compose(self.measure_circ(reg_a,reg_b,reg_c,num_features=num_features),inplace=True)

        o1 = SparsePauliOp.from_list([("I"*2*num_features+"X"*num_features,1)])

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=list(f1.parameters)+list(f2.parameters),
            weight_params=pqc.parameters,
            observables = o1,
            input_gradients = True
        )

        return qnn

    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True).double()
        self.qnn = TorchConnector(self.get_QNN(5)).double()

    def forward(self, x):
        m, s, e = x.shape

        t = torch.empty((0,s,2*e),dtype=torch.double)
        x = x.double()

        #print(x.shape)
        for bat in x:
            for vec in normalize(bat):
                #print(vec)
                #print((vec.repeat(bat.shape[0],1).double(),normalize(bat)).double())
                t = torch.cat((t,torch.stack((vec.repeat(bat.shape[0],1),normalize(bat)),1).reshape(1,s,2*e)))
                #print(t)
                #print("ppppppppppppppppp")
                #break
        
        
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

        #print(t.reshape([-1,2*e])[:,-32:].norm(dim=1),t.type())
        #print(t.reshape([-1,2*e])[:,:-32].norm(dim=1),t.type())
        x_attention = self.qnn(t.reshape([-1,2*e]))
        x_attention = x_attention.reshape(m,s,s)
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
