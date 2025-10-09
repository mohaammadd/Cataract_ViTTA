import torch
import torch.nn.functional as F
import torch.optim as optim
import random

class ViTTAAdapter:
    """
    Video Test-Time Adaptation (ViTTA)
    - Aligns feature statistics with source domain
    - Uses temporal augmentation + prediction consistency
    """

    def __init__(self, model, source_stats, lr=1e-5, lambda_cons=0.1, alpha=0.1,
                 layers=("last",), sampling="uniform", device="cuda"):
        self.model = model
        self.source_stats = source_stats   # {layer: (mu_src, var_src)}
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_cons = lambda_cons
        self.alpha = alpha
        self.device = device
        self.layers = layers
        self.sampling = sampling

        # Initialize EMA stats per layer
        self.ema_stats = {
            l: (torch.zeros_like(source_stats[l][0]),
                torch.ones_like(source_stats[l][1]))
            for l in layers
        }

    # ------------------------------
    # EMA update
    # ------------------------------
    def update_stats(self, feats, layer):
        mu = feats.mean(0)
        # Use unbiased=False to avoid nan for batch size 1
        if feats.shape[0] == 1:
            var = torch.zeros_like(mu)
        else:
            var = feats.var(0, unbiased=False)
        mu_ema, var_ema = self.ema_stats[layer]
        mu_new = self.alpha * mu + (1 - self.alpha) * mu_ema
        var_new = self.alpha * var + (1 - self.alpha) * var_ema
        self.ema_stats[layer] = (mu_new, var_new)
        return mu_new, var_new

    # ------------------------------
    # Alignment loss
    # ------------------------------
    def loss_alignment(self, mu_new, var_new, mu_src, var_src):
        return torch.norm(mu_new - mu_src, 1) + torch.norm(var_new - var_src, 1)

    # ------------------------------
    # Temporal view sampling
    # ------------------------------
    def sample_views(self, x, num_views=2):
        B, T, C, H, W = x.shape
        views = []
        for _ in range(num_views):
            if self.sampling == "uniform":
                idx = torch.linspace(0, T - 1, T // 2).long()
            elif self.sampling == "random":
                idx = torch.sort(torch.randint(0, T, (T // 2,)))[0]
            else:
                raise ValueError(f"Unknown sampling: {self.sampling}")
            views.append(x[:, idx, :, :, :])
        return views

    # ------------------------------
    # Adaptation step
    # ------------------------------
    def adapt_step(self, x, num_views=2):
        self.model.train()
        logits_list, feats_dict_list = [], []

        # Create multiple temporal views
        for view in self.sample_views(x, num_views=num_views):
            logits, feats = self.model(view)
            logits_list.append(F.softmax(logits, dim=1))
            feats_dict_list.append({"last": feats})  # extendable: more layers

        # Pseudo-label = avg prediction
        pseudo = sum(logits_list) / len(logits_list)

        # Consistency loss
        loss_cons = sum(F.l1_loss(log, pseudo.detach()) for log in logits_list)

        # Alignment loss (over selected layers)
        loss_align = 0
        for layer in self.layers:
            mu_new, var_new = self.update_stats(feats_dict_list[0][layer], layer)
            mu_src, var_src = self.source_stats[layer]
            loss_align += self.loss_alignment(mu_new, var_new, mu_src, var_src)

        # Total loss
        loss = loss_align + self.lambda_cons * loss_cons

        # One gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"total": loss.item(), "align": loss_align.item(), "cons": loss_cons.item()}
