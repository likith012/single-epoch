from torch import nn
from resnet1d import BaseNet
import torch

def sleep_model(n_channels, input_size_samples, n_dim = 256):
    class attention(nn.Module):
        
        def __init__(self, n_dim):
            super(attention,self).__init__()
            self.att_dim = n_dim
            self.W = nn.Parameter(torch.randn(n_dim, self.att_dim))
            self.V = nn.Parameter(torch.randn(self.att_dim, 1))
            self.scale = self.att_dim**-0.5
            
        def forward(self,x):
            x = x.permute(0, 2, 1)
            e = torch.matmul(x, self.W)
            e = torch.matmul(torch.tanh(e), self.V)
            e = e*self.scale
            n1 = torch.exp(e)
            n2 = torch.sum(torch.exp(e), 1, keepdim=True)
            alpha = torch.div(n1, n2)
            x = torch.sum(torch.mul(alpha, x), 1)
            return x
        
    class encoder(nn.Module):

        def __init__(self, n_channels, n_dim):
            super(encoder,self).__init__()
            self.model = BaseNet(input_channel = n_channels)
            self.attention = attention(n_dim)
            
        def forward(self, x): 
            x = self.model(x)
            x = self.attention(x)
            return x
        
    class Net(nn.Module):
        
        def __init__(self, n_channels, n_dim):
            super().__init__()
            self.enc = encoder(n_channels, n_dim)
            self.n_dim = n_dim
            
            self.p1 = nn.Sequential(
                nn.Linear(self.n_dim, self.n_dim // 2, bias=True),
                # nn.BatchNorm1d(self.n_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_dim // 2, self.n_dim // 2, bias=True),
            )

            self.p2 = nn.Sequential(
                nn.Linear(self.n_dim, self.n_dim // 2, bias=True),
                # nn.BatchNorm1d(self.n_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_dim // 2, self.n_dim // 2, bias=True),
            )
           
        def forward(self, x, proj='mid'):
            x = self.enc(x)
            
            if proj == 'top':
                x = self.p1(x)
                return x
            elif proj == 'bottom':
                x = self.p2(x)
                return x
            elif proj == 'mid':
                return x
            else:
                raise Exception("Fix the projection heads")
            
    return Net(n_channels, n_dim)
