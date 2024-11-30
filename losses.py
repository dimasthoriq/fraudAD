import torch


class SupConLoss(torch.nn.Module):
    """
    Supervised contrastive loss as introduced in the SSD paper.
    Also supports the unsupervised contrastive loss in SimCLR paper.
    """

    def __init__(self, temperature=0.07, contrast_mode='one'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features):
        """
        Compute loss for model.
        Equals to SimCLR unsupervised contrastive loss when `labels` and `mask` are both None.

        Args:
            features: hidden vector of shape [batch_size, n_views, ...].

        Returns:
            A loss scalar
        """
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown contrast mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SADLoss(torch.nn.Module):
    def __init__(self, eta=1.0, epsilon=1e-6, method='sad'):
        super(SADLoss, self).__init__()
        self.eta = eta
        self.epsilon = epsilon
        self.method = method

    def forward(self, z, y, c, cov):
        if self.method == 'sad-maha':
            delta = z - c
            dist = torch.sum(delta @ torch.linalg.pinv(cov) * delta, dim=-1)
        else:
            dist = torch.sum((z - c) ** 2, dim=1)
        losses = torch.where(y == 0, dist, self.eta * (dist+self.epsilon)**-y)
        return torch.mean(losses)
