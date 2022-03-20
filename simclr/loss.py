import torch
import torch.nn.functional as F

# Loss function
class loss_fn(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=1.0):
        """
        T: softmax temperature (default: 0.07)
        """
        super(loss_fn, self).__init__()
        self.T = T
        self.device = device

    def forward(self, anc, pos, neg):

        # L2 normalize
        anc = F.normalize(anc, p=2, dim=1)  # B, 128
        pos = F.normalize(pos, p=2, dim=1)  # B, 128
        neg = F.normalize(neg, p=2, dim=1)  # B, 128

        pos_numerator = torch.exp((anc*pos).sum(axis=-1)/self.T) # B
        pos_numerator = torch.cat([pos_numerator,pos_numerator],dim=0)
        neg = neg.permute(1,0) # 128, B
        neg1_denominator = torch.exp((torch.mm(torch.cat([anc,pos],dim=0), neg).sum(axis=-1))/self.T) # B
       
        loss = -torch.log(pos_numerator/(pos_numerator+neg1_denominator))
        return loss.mean()