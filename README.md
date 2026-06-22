# HRC-Net

HRC-Net is a PyTorch reimplementation scaffold of the method described in
`HRC-Net_Revised_v5.docx`, organized by borrowing the project layout ideas from
the original `ShellNet` repository.

## Project Layout

- `data/`: dataset download and preprocessing entry scripts
- `configs/`: experiment configuration files for three benchmarks
- `models/`: HRC-Net backbone, HRConv operator, attention, and shared layers
- `utils/`: data loading, training, evaluation, and metrics helpers
- `scripts/`: one-command training and evaluation scripts
- `checkpoints/`: placeholder locations for pretrained weights

## Method Mapping

This scaffold keeps the same high-level workflow as ShellNet:

1. Dataset-specific configs define paths and hyper-parameters.
2. A shared training loop loads config, model, data, optimizer, and metrics.
3. Task-specific heads are built on top of a reusable point backbone.

The core HRC-Net modules follow the paper description:

- `models/hrconv.py`: hierarchical radius convolution with band partition,
  distance-aware weighting, positional offset, and residual fusion
- `models/attention.py`: lightweight channel attention after multi-band fusion
- `models/hrcnet.py`: encoder plus classification / part segmentation heads

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate hrc-net
pip install -r requirements.txt
```

Train on ModelNet40:

```bash
bash scripts/train_modelnet40.sh
```

Train on ScanObjectNN:

```bash
bash scripts/train_scanobjectnn.sh
```

Train on ShapeNetPart:

```bash
bash scripts/train_shapenetpart.sh
```

Evaluate all configured checkpoints:

```bash
bash scripts/eval_all.sh
```

## Notes

- The repository currently provides a faithful code structure and functional
  module scaffold for HRC-Net reproduction.
- Files under `checkpoints/` are placeholders only. Replace them with real
  `.pth` weights before running evaluation commands.
- Dataset download scripts are organized for reproducibility and can be adapted
  to local mirrors if official links change.
