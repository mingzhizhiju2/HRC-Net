import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SharedMLP, Dense
from .hrconv import HRConv


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = xyz.new_zeros(B, npoint, dtype=torch.long)
    distance = xyz.new_ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, device=xyz.device, dtype=torch.long)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1).indices

    return centroids


def gather_points(xyz, idx):
    B, N, _ = xyz.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device).view(view_shape).repeat(repeat_shape)
    return xyz[batch_indices, idx, :]


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, K, band_edges, sample_count, reduction=4):
        super().__init__()
        self.sample_count = sample_count
        self.hrconv = HRConv(in_channels, out_channels, K, band_edges, reduction)

    def forward(self, pts, fts):
        if self.sample_count < pts.shape[1]:
            idx = farthest_point_sample(pts, self.sample_count)
            qrs = gather_points(pts, idx)
        else:
            qrs = pts
        out_fts = self.hrconv(pts, fts, qrs)
        return qrs, out_fts


class HRCNetCls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        enc_cfg = config["model"]["encoder"]
        stem = config["model"]["stem_channels"]
        clf_cfg = config["model"]["classifier"]
        attn_red = config["model"]["attention_reduction"]

        self.stem = nn.Sequential(
            nn.Conv1d(config["model"]["input_channels"], stem, 1, bias=False),
            nn.BatchNorm1d(stem),
            nn.ReLU(inplace=True),
            nn.Conv1d(stem, stem, 1, bias=False),
            nn.BatchNorm1d(stem),
            nn.ReLU(inplace=True),
        )

        in_ch = stem
        self.encoder = nn.ModuleList()
        for stage_cfg in enc_cfg:
            self.encoder.append(
                EncoderStage(
                    in_channels=in_ch,
                    out_channels=stage_cfg["out_channels"],
                    K=stage_cfg["k"],
                    band_edges=stage_cfg["band_edges"],
                    sample_count=stage_cfg["sample_count"],
                    reduction=attn_red,
                )
            )
            in_ch = stage_cfg["out_channels"]

        clf_layers = []
        prev_c = in_ch
        for hc in clf_cfg["hidden_channels"]:
            clf_layers.append(Dense(prev_c, hc, dropout=clf_cfg.get("dropout", 0)))
            prev_c = hc
        self.classifier = nn.Sequential(*clf_layers)
        self.logits = nn.Linear(prev_c, self.num_classes)

    def forward(self, pts):
        fts = self.stem(pts.permute(0, 2, 1))
        for stage in self.encoder:
            pts, fts = stage(pts, fts)
        fts = fts.max(dim=-1).values
        fts = self.classifier(fts)
        return self.logits(fts)


class HRCNetPartSeg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        self.num_categories = config["num_categories"]
        enc_cfg = config["model"]["encoder"]
        dec_cfg = config["model"]["decoder"]
        seg_cfg = config["model"]["segmentation_head"]
        stem = config["model"]["stem_channels"]
        attn_red = config["model"]["attention_reduction"]

        self.category_embed = nn.Embedding(self.num_categories, 64)

        self.stem = nn.Sequential(
            nn.Conv1d(config["model"]["input_channels"], stem, 1, bias=False),
            nn.BatchNorm1d(stem),
            nn.ReLU(inplace=True),
            nn.Conv1d(stem, stem, 1, bias=False),
            nn.BatchNorm1d(stem),
            nn.ReLU(inplace=True),
        )

        in_ch = stem
        self.encoder = nn.ModuleList()
        self.encoder_pts = []
        self.encoder_fts = []
        for stage_cfg in enc_cfg:
            self.encoder.append(
                EncoderStage(
                    in_channels=in_ch,
                    out_channels=stage_cfg["out_channels"],
                    K=stage_cfg["k"],
                    band_edges=stage_cfg["band_edges"],
                    sample_count=stage_cfg["sample_count"],
                    reduction=attn_red,
                )
            )
            in_ch = stage_cfg["out_channels"]

        dec_layers = []
        prev_c = in_ch
        for dc in dec_cfg["channels"]:
            dec_layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_c, dc, 1, bias=False),
                    nn.BatchNorm1d(dc),
                    nn.ReLU(inplace=True),
                )
            )
            prev_c = dc

        self.decoder = nn.ModuleList(dec_layers)
        self.seg_head = nn.Sequential(
            nn.Conv1d(prev_c + 64, seg_cfg["hidden_channels"], 1, bias=False),
            nn.BatchNorm1d(seg_cfg["hidden_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(seg_cfg.get("dropout", 0.3)),
            nn.Conv1d(seg_cfg["hidden_channels"], self.num_classes, 1),
        )

    def forward(self, pts, categories=None):
        fts = self.stem(pts.permute(0, 2, 1))
        skip_pts = [pts]
        skip_fts = [fts]

        for stage in self.encoder:
            pts, fts = stage(pts, fts)
            skip_pts.append(pts)
            skip_fts.append(fts)

        x = fts
        for i, dec in enumerate(self.decoder):
            sp = skip_pts[-(i + 2)]
            sf = skip_fts[-(i + 2)]
            x = F.interpolate(x, size=sf.shape[-1], mode="nearest")
            x = dec(x)

        if categories is not None:
            cat_feat = self.category_embed(categories)
            cat_feat = cat_feat.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, cat_feat], dim=1)

        x = self.seg_head(x)
        return x.permute(0, 2, 1).contiguous()


def build_model(config):
    name = config["model"]["name"]
    if name == "hrcnet_cls":
        return HRCNetCls(config)
    elif name == "hrcnet_partseg":
        return HRCNetPartSeg(config)
    else:
        raise ValueError(f"Unknown model name: {name}")
