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

        pos_num = torch.exp((anc*pos).sum(axis=-1)/self.T) # B
        pos_num= torch.cat([pos_num,pos_num],dim=0) #2B

        pos_denom = torch.exp(torch.mm(anc,pos.permute(1,0))/self.T) #B,B

        mask = ~torch.eye(pos_denom.shape[0],device=self.device).bool()
        pos_denominator1 = pos_denom.masked_select(mask).view(pos_denom.shape[0],-1).sum(dim=-1) #B,B-1
        pos_denominator2 = pos_denom.t().masked_select(mask).view(pos_denom.shape[0],-1).sum(dim=-1) #B,B-1
        pos_denominator = torch.cat([pos_denominator1,pos_denominator2],dim=0) #2B

        neg = neg.permute(1,0) # 128, B
        neg1_denominator = torch.exp((torch.mm(torch.cat([anc,pos],dim=0), neg).sum(axis=-1))/self.T) # 2B
       
        loss = -torch.log(pos_num/(pos_num+pos_denominator+neg1_denominator))
        return loss.mean()
