import torch
import torch.nn as nn


def knn_points(q, k, pts):
    dist = torch.cdist(q, pts)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=True).indices
    return idx


class HRConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, band_edges, reduction=4):
        super().__init__()
        self.K = K
        self.B = len(band_edges)
        self.band_edges = band_edges

        self.mlp_multi_feat = nn.Sequential(
            nn.Conv2d(9, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        per_band_in = out_channels + in_channels if in_channels > 0 else out_channels
        self.per_band_convs = nn.ModuleList()
        for _ in range(self.B):
            self.per_band_convs.append(
                nn.Sequential(
                    nn.Conv2d(per_band_in, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.distance_weight_mlp = nn.Sequential(
            nn.Conv2d(1, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, 1),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * self.B, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attention_fc = nn.Sequential(
            nn.Linear(out_channels, max(out_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(out_channels // reduction, 8), out_channels, bias=False),
            nn.Sigmoid(),
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, pts, fts, qrs):
        B, P, _ = qrs.shape
        idx = knn_points(qrs, self.K, pts)

        nn_pts = torch.gather(
            pts.unsqueeze(1).expand(-1, P, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        center = qrs.unsqueeze(2).expand(-1, -1, self.K, -1)

        multi_feat = torch.cat(
            [
                nn_pts - center,
                nn_pts,
                center,
            ],
            dim=-1,
        )
        multi_feat = multi_feat.permute(0, 3, 1, 2)
        f_NEC = self.mlp_multi_feat(multi_feat)

        if fts is not None:
            fts_knn = torch.gather(
                fts.unsqueeze(2).expand(-1, -1, self.K, -1),
                1,
                idx.unsqueeze(-1).expand(-1, -1, -1, fts.shape[-1]),
            )
            fts_knn = fts_knn.permute(0, 3, 1, 2)
            conv_in = torch.cat([f_NEC, fts_knn], dim=1)
        else:
            conv_in = f_NEC

        dist = torch.sqrt(torch.sum((nn_pts - center) ** 2, dim=-1) + 1e-8)
        dist_max = dist.max(dim=-1, keepdim=True).values
        dist_norm = dist / (dist_max + 1e-8)

        dist_attn = self.distance_weight_mlp(dist_norm.unsqueeze(1))
        dist_attn = torch.softmax(dist_attn, dim=-1)

        band_outputs = []
        for b in range(self.B):
            low = 0.0 if b == 0 else self.band_edges[b - 1]
            high = self.band_edges[b]
            mask = ((dist_norm >= low) & (dist_norm < high)).float()
            mask_expanded = mask.unsqueeze(1)
            masked = conv_in * mask_expanded * dist_attn
            feat_b = self.per_band_convs[b](masked)
            band_outputs.append(feat_b)

        band_outputs = torch.cat(band_outputs, dim=1)
        band_outputs = band_outputs.max(dim=-1).values
        fused = self.fusion(band_outputs)

        y = self.avg_pool(fused).view(B, -1)
        attn = self.attention_fc(y).view(B, -1, 1)
        out = fused * attn

        if fts is not None:
            fts_pooled = torch.gather(
                fts.unsqueeze(2).expand(-1, -1, self.K, -1),
                1,
                idx.unsqueeze(-1).expand(-1, -1, -1, fts.shape[-1]),
            )
            fts_pooled = fts_pooled.mean(dim=2).permute(0, 2, 1).contiguous()
            shortcut = self.shortcut(fts_pooled)
        else:
            shortcut = 0

        return out + shortcut