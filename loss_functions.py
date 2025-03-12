import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature_init=0.07, device="cuda"):
        super(ContrastiveLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))
        self.clip_loss = nn.CrossEntropyLoss()
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
        loss_i = self.clip_loss(similarity_matrix, labels) # im -> txt
        loss_t = self.clip_loss(similarity_matrix.T, labels) # txt -> im

        return (loss_i + loss_t) / 2, similarity_matrix

class SigLipLoss(nn.Module):
    def __init__(self, temperature_init=0.1, device="cuda"):
        super(SigLipLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init))) #log10 as per paper
        self.b = nn.Parameter(torch.tensor([-10.0])) # -10 as per paper
        self.sig_loss = nn.LogSigmoid()
        self.device = device

    def forward(self, feats_one, feats_two):

        feats_one = nn.functional.normalize(feats_one, dim=1, p=2.0)
        feats_two = nn.functional.normalize(feats_two, dim=1, p=2.0)

        similarity_matrix = torch.matmul(feats_one, feats_two.T) * self.logit_scale.exp() + self.b

        labels = 2 * torch.eye(similarity_matrix.size(0)).to(self.device) - torch.ones(similarity_matrix.size(0)).to(self.device)

        loss = -torch.sum(self.sig_loss(labels * similarity_matrix))/similarity_matrix.size(0)

        return loss, similarity_matrix
