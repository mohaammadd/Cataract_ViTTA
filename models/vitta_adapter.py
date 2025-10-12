import torch
import torch.nn.functional as F
import torch.optim as optim
import random


class ViTTAAdapter:
    """
    ViTTA: Video Test-Time Adaptation
    Implements temporal and appearance augmentations as in the original paper.
    """

    def __init__(self, model, source_stats, lr=1e-5, lambda_cons=0.1, alpha=0.1,
                 layers=("last",), sampling="random", device="cuda", num_subvideo_frames=8):
        self.model = model
        self.source_stats = source_stats
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_cons = lambda_cons
        self.alpha = alpha
        self.device = device
        self.layers = layers
        self.sampling = sampling
        self.num_subvideo_frames = num_subvideo_frames

        # Initialize EMA stats for each selected layer
        self.ema_stats = {
            l: (torch.zeros_like(source_stats[l][0]),
                torch.ones_like(source_stats[l][1]))
            for l in layers
        }

    # ------------------------------
    # EMA Update for feature stats
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
    # Feature alignment loss (ViTTA)
    # ------------------------------
    def loss_alignment(self, mu_new, var_new, mu_src, var_src):
        return torch.norm(mu_new - mu_src, 1) + torch.norm(var_new - var_src, 1)

    # ------------------------------
    # Appearance augmentations (ViTTA)
    # ------------------------------
    def augment_view(self, view):
        # view: [B, T, C, H, W]
        # Gaussian noise
        if random.random() < 0.3:
            noise = 0.02 * torch.randn_like(view)
            view = torch.clamp(view + noise, 0, 1)

        # Horizontal flip
        if random.random() < 0.3:
            view = torch.flip(view, dims=[-1])

        # Brightness & contrast jitter
        if random.random() < 0.3:
            scale = 1.0 + 0.15 * torch.randn(1, device=view.device).item()
            bias = 0.02 * torch.randn(1, device=view.device).item()
            view = torch.clamp(view * scale + bias, 0, 1)

        # Gaussian blur (simulated with average pooling)
        if random.random() < 0.1:
            kernel = torch.ones((1, 1, 3, 3), device=view.device) / 9.0
            B, T, C, H, W = view.shape
            view = view.view(-1, 1, H, W)
            view = F.conv2d(view, kernel, padding=1)
            view = view.view(B, T, C, H, W)

        return view

    # ------------------------------
    # Temporal augmentations
    # ------------------------------
    def sample_views(self, x, num_views=4):
        B, T, C, H, W = x.shape
        views = []
        n = min(self.num_subvideo_frames, T)
        for _ in range(num_views):
            if self.sampling == "uniform":
                idx = torch.linspace(0, T - 1, n).long()
            elif self.sampling == "random":
                idx = torch.sort(torch.randint(0, T, (n,)))[0]
            else:
                raise ValueError(f"Unknown sampling type: {self.sampling}")

            view = x[:, idx, :, :, :]
            view = self.augment_view(view)
            views.append(view)
        return views

    # ------------------------------
    # Adaptation step
    # ------------------------------
    def adapt_step(self, x, num_views=4):
        self.model.train()
        logits_list, feats_list = [], []

        # Generate augmented temporal views
        for view in self.sample_views(x, num_views=num_views):
            logits, feats = self.model(view)
            logits_list.append(F.softmax(logits, dim=1))
            feats_list.append({"last": feats})

        # Pseudo-label (mean of predictions)
        pseudo = sum(logits_list) / len(logits_list)

        # Consistency loss between all views
        loss_cons = sum(F.l1_loss(log, pseudo.detach()) for log in logits_list)

        # Alignment loss using EMA stats
        loss_align = 0
        for layer in self.layers:
            mu_new, var_new = self.update_stats(feats_list[0][layer], layer)
            mu_src, var_src = self.source_stats[layer]
            loss_align += self.loss_alignment(mu_new, var_new, mu_src, var_src)

        # Total loss = alignment + consistency
        loss = loss_align + self.lambda_cons * loss_cons

        # One optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "total": loss.item(),
            "align": loss_align.item(),      
            "cons": loss_cons.item(),
        }
