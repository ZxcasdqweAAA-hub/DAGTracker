import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive=2.0):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive

    def forward(self, p_rela, y):

        positive_loss = y * torch.log(p_rela + 1e-10)
        positive_loss = -self.weight_positive * positive_loss

        negative_loss = (1 - y) * torch.log(1 - p_rela + 1e-10)
        negative_loss = -negative_loss

        total_loss = torch.mean(positive_loss + negative_loss)

        return total_loss


class CircleLoss_(nn.Module):
    def __init__(self, m: float = 0.25, gamma: float = 256) -> None:
        super(CircleLoss_, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        sim_mat = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        mask_self = ~torch.eye(label_mask.size(0), dtype=torch.bool, device=label_mask.device)
        label_mask = label_mask & mask_self

        S_p = sim_mat[label_mask]
        S_n = sim_mat[~label_mask]

        if S_p.numel() == 0 or S_n.numel() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        O_p = 1 + self.m
        O_n = -self.m

        delta_p = 1 - self.m
        delta_n = self.m

        alpha_p = torch.clamp_min(O_p - S_p.detach(), min=0)
        alpha_n = torch.clamp_min(S_n.detach() - O_n, min=0)

        term_p = self.gamma * alpha_p * (S_p - delta_p)
        term_n = self.gamma * alpha_n * (delta_n - S_n)

        loss = self.soft_plus(torch.logsumexp(term_p, dim=0) + torch.logsumexp(term_n, dim=0))

        return loss





class CircleLoss(Module):
    """Implementation for "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    Note: this is the classification based implementation of circle loss.
    """
    def __init__(self, feat_dim, num_class, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.gamma = gamma

        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1-margin
        self.delta_n = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index_pos = torch.zeros_like(cos_theta)
        index_pos.scatter_(1, labels.data.view(-1, 1), 1)
        index_pos = index_pos.byte().bool()
        index_neg = torch.ones_like(cos_theta)
        index_neg.scatter_(1, labels.data.view(-1, 1), 0)
        index_neg = index_neg.byte().bool()

        alpha_p = torch.clamp_min(self.O_p - cos_theta.detach(), min=0.)
        alpha_n = torch.clamp_min(cos_theta.detach() - self.O_n, min=0.)

        logit_p = alpha_p * (cos_theta - self.delta_p)
        logit_n = alpha_n * (cos_theta - self.delta_n)

        output = cos_theta * 1.0
        output[index_pos] = logit_p[index_pos]
        output[index_neg] = logit_n[index_neg]
        output *= self.gamma
        return output


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size,)  # 修正注释：原注释错误
        """
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(mat1=inputs, mat2=inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask.fill_diagonal_(0)

        dist_ap, dist_an = [], []
        for i in range(n):
            pos_dist = dist[i][mask[i]]
            neg_dist = dist[i][~mask[i]]

            if len(pos_dist) > 0 and len(neg_dist) > 0:
                dist_ap.append(pos_dist.max().unsqueeze(0))  # hardest positive
                dist_an.append(neg_dist.min().unsqueeze(0))  # hardest negative

        if len(dist_ap) == 0 or len(dist_an) == 0:
            loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
            if self.mutual:
                return loss, dist
            return loss
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        if self.mutual:
            return loss, dist
        return loss



class SoftmaxFocalLoss(nn.Module):
    """
    Row-wise softmax + focal loss
    """
    def __init__(self, gamma=2.0, tau=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, target, N, M):
        """
        logits: (N*M, 1)  raw matching logits
        target: (N*M, 1)  binary labels
        """

        # ---- reshape to (N, M) ----
        logits = logits.view(N, M)
        target = target.view(N, M)

        # ---- row-wise softmax (competition among candidates) ----
        prob = F.softmax(logits / self.tau, dim=1)  # (N, M)

        # ---- focal loss ----
        pt = prob * target + (1 - prob) * (1 - target)
        loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist

def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss


def circle_pair_bce(
    emb, labels,
    m=0.25, gamma=64
):
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    sim = torch.matmul(emb, emb.t())

    label_mat = labels[:, None] == labels[None, :]
    label_mat = label_mat.float()

    alpha_p = torch.clamp_min(1 + m - sim.detach(), 0.)
    alpha_n = torch.clamp_min(sim.detach() + m, 0.)

    logit_p = alpha_p * (sim - (1 - m))
    logit_n = alpha_n * (sim - m)

    logits = torch.where(label_mat > 0, logit_p, logit_n)
    logits = gamma * logits

    loss = F.binary_cross_entropy_with_logits(
        logits, label_mat
    )
    return loss


def pairwise_circleloss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)

    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss