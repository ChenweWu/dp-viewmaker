import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MFocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction="None"):
        super(MFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
         targets = targets.float()
         if inputs.dim()>2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)
         
         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
         p_t = torch.exp(-ce_loss)
         loss = ce_loss * ((1 - p_t) ** self.gamma)
         if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
         if self.reduction == "mean":
            loss = loss.mean()
         elif self.reduction == "sum":
            loss = loss.sum()

         return 10*loss
'''
    def forward(self, input, target):
         if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
         target = target.view(-1,1)
         logpt = F.log_softmax(input)
         logpt = logpt.gather(1,target.to(torch.int64))
         logpt = logpt.view(-1)
         pt = Variable(logpt.data.exp())
         if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0,target.data.view(-1))
                logpt = logpt * Variable(at)
         loss = -1 * (1-pt)**self.gamma * logpt

         if self.size_average:
            return loss.mean()
         else:
            return loss.sum()
    '''
