import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature_init=0.07, device="cuda"):
        super(ContrastiveLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))
        self.device = device

    def forward(self, feats_one, feats_two):

        # normalize for vecotrs for cos similarity (possibly equivalent: add a layer norm after projection)
        feats_one = nn.functional.normalize(feats_one, dim=1, p=2.0)
        feats_two = nn.functional.normalize(feats_two, dim=1, p=2.0)

        # similarity matrix
        similarity_matrix = torch.matmul(feats_one, feats_two.T) * self.logit_scale.exp()

        # constrative learning labels
        labels = torch.arange(similarity_matrix.size(0)).to(self.device)
        
        # cel loss
        loss_i = nn.CrossEntropyLoss()(similarity_matrix, labels)
        loss_t = nn.CrossEntropyLoss()(similarity_matrix.T, labels)

        return (loss_i + loss_t) / 2, similarity_matrix

class SigLipLoss(nn.Module):
    def __init__(self, temperature_init=0.07, device="cuda"):
        super(SigLipLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))
        self.device = device

    def forward(self, feats_one, feats_two):

        #TODO

        return 0
