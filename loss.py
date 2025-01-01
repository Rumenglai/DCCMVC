import torch
import torch.nn as nn
import math
import sys
import probtorch
from probtorch.util import expand_inputs

def elbo(q, pA, pB, lamb1=1.0, lamb2=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):

    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images1_sharedA'],
                                                               latents=['privateA', 'sharedA'], sample_dim=0,
                                                               batch_dim=1,
                                                               beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images2_sharedB'],
                                                               latents=['privateB', 'sharedB'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta2, bias=bias)
    reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images1_poe'],
                                                                     latents=['privateA', 'poe'], sample_dim=0,
                                                                     batch_dim=1,
                                                                     beta=beta1, bias=bias)
    reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images2_poe'],
                                                                     latents=['privateB', 'poe'], sample_dim=0,
                                                                     batch_dim=1,
                                                                     beta=beta2, bias=bias)
    reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images1_sharedB'],
                                                                   latents=['privateA', 'sharedB'], sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=beta1, bias=bias)
    reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['images2_sharedA'],
                                                                   latents=['privateB', 'sharedA'], sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=beta2, bias=bias)


    reconst_loss_poeA = torch.tensor(0)
    reconst_loss_poeB = torch.tensor(0)

    loss = (lamb1 * reconst_loss_A - kl_A) + (lamb2 * reconst_loss_B - kl_B) + \
            (lamb1 * reconst_loss_crA - kl_crA) + (lamb2 * reconst_loss_crB - kl_crB)  + \
    (lamb1 * reconst_loss_poeA - kl_poeA) + (lamb2 * reconst_loss_poeB - kl_poeB)

    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB]

def compute_joint(view1, view2):
    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    p_i_j = p_i_j / p_i_j.sum()

    return p_i_j
def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()
    return loss
class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


