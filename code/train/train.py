"""HQ-Edit forward/inverse feature training for FlowCIR v1."""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import HQEDIT_FEATURE_DIR, SAVE_ROOT, CLIP_MODEL, CLIP_DIMS
from models.fm_condition import DeepFlowMatchingNet
from hq_dataset.hqEdit_dataset_feat_inverse import HQEditFeatShardDataset

def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seed for python / numpy / torch (cpu & cuda).

    Args:
        seed: random seed
        deterministic: if True, make cudnn deterministic (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

def seed_worker(worker_id: int):
    """
    Make dataloader worker seed deterministic.
    This will be called at the beginning of each worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_linear_ramp(epoch: int, warmup_epochs: int, start: float, end: float) -> float:
    """Linearly ramp from start to end over warmup_epochs. After warmup, clamp to end."""
    if warmup_epochs <= 0:
        return end
    p = min(max(epoch / float(warmup_epochs), 0.0), 1.0)
    return start + p * (end - start)

def get_infonce_weight(epoch: int, args) -> float:
    if epoch > args.infonce_start_epochs:
        return get_linear_ramp(
            epoch=epoch-args.infonce_start_epochs,
            warmup_epochs=args.infonce_warmup_epochs,
            start=args.infonce_start_weight,
            end=args.infonce_max_weight,
        )
    else:
        return 0

def info_nce_topk_hardneg_image_retrieval(
    transferred_features, target_image_feat, temperature=0.07,
    normalize=True, k=32, exclude_too_hard=False, eps=0.02,
):
    """Sampled CE over the positive and top-k negatives.

    eps retains the original implementation's temperature-scaled logit units.
    When fewer than k semi-hard negatives remain, use unfiltered negatives.
    A singleton batch has no negatives and contributes zero contrastive loss.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if transferred_features.ndim != 2 or transferred_features.shape != target_image_feat.shape:
        raise ValueError("Expected matching [batch, dim] feature tensors")
    if transferred_features.size(0) < 2:
        return transferred_features.sum() * 0.0
    with autocast(enabled=False):
        q = transferred_features.float()
        target = target_image_feat.float()
        if normalize:
            q = F.normalize(q, dim=-1)
            target = F.normalize(target, dim=-1)
        logits = (q @ target.t()) / temperature
        batch_size = logits.size(0)
        k = max(1, min(int(k), batch_size - 1))
        pos = logits.diagonal()
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        negatives = logits.masked_fill(diagonal, float('-inf'))
        if exclude_too_hard:
            filtered = negatives.masked_fill(negatives >= pos[:, None] - eps, float('-inf'))
            enough = torch.isfinite(filtered).sum(dim=1) >= k
            negatives = torch.where(enough[:, None], filtered, negatives)
        hard_values = negatives.topk(k, dim=1).values
        sampled_logits = torch.cat((pos[:, None], hard_values), dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        return F.cross_entropy(sampled_logits, labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='control the number of training data')
    parser.add_argument('--dim', type=int, default=CLIP_DIMS[CLIP_MODEL], help='dim')
    parser.add_argument('--data_dir', type=str, default=str(HQEDIT_FEATURE_DIR), help='control the number of training data')
    parser.add_argument("--num_res_blocks", type=int, default=8, help="number of residual blocks in DeepFlowMatchingNet")
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--save_dir', type=str, default=str(SAVE_ROOT / "v1"), help='directory to save the trained model')
    parser.add_argument('--lambda_infoNce', type=float, default=0.0, help='weight for infoNce loss')
    parser.add_argument('--dataset', type=str, default='hqedit', choices=['hqedit'], help='HQ-Edit features')
    parser.add_argument('--negK', type=int, default=512, help='number of hard negatives for infoNce loss')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--deterministic', action='store_true', default=True, help='enable deterministic training for better reproducibility')
    parser.add_argument('--no_deterministic', action='store_false', dest='deterministic', help='disable deterministic training for speed')
    parser.add_argument('--exclude_too_hard', action='store_true', help='whether to exclude too hard negatives in infoNce loss')
    parser.add_argument('--eps', type=float, default=0.05, help='margin for excluding too hard negatives in infoNce loss')
    parser.add_argument("--infonce_warmup_epochs", type=int, default=20,
                        help="Linearly ramp InfoNCE weight for these epochs.")
    parser.add_argument("--infonce_max_weight", type=float, default=None,
                        help="Max value of InfoNCE weight after warmup.")
    parser.add_argument("--infonce_start_weight", type=float, default=0.0,
                        help="Start value of InfoNCE weight at epoch 0.")
    parser.add_argument("--infonce_start_epochs", type=int, default=0,
                        help="Epoch to start ramping up InfoNCE weight. Before this, weight is 0.")

    parser.add_argument("--topk_start_epoch", type=int, default=0,
                        help="Switch the semi-hard logit margin from eps_pre to eps at this zero-based epoch.")
    parser.add_argument("--eps_pre", type=float, default=0.05,
                        help="Semi-hard logit margin before topk_start_epoch.")

    parser.add_argument("--exclude_too_hard_until_epoch", type=int, default=0,
                        help="For epochs < this, set exclude_too_hard=True (safer). After that, use args.exclude_too_hard.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    if args.infonce_max_weight is None:
        args.infonce_max_weight = args.lambda_infoNce
    if args.batch_size < 1 or args.epochs < 1 or args.n < 0:
        parser.error("batch_size and epochs must be positive; n must be nonnegative")
    device = torch.device(args.device)
    scaler = GradScaler(enabled=device.type == "cuda")

    print(args)
    set_seed(args.seed, deterministic=args.deterministic)
    if args.n == 0:
        args.n = None

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = HQEditFeatShardDataset(
        feat_dir=args.data_dir, dtype=torch.float32, device="cpu", shard_limit=args.n,
    )
    if len(dataset) == 0:
        raise ValueError("The training dataset is empty")
    if dataset.input_image_feat.shape[1] != args.dim:
        raise ValueError(f"Feature dimension {dataset.input_image_feat.shape[1]} != --dim {args.dim}")
    print(f"Dataset size: {len(dataset)} samples")

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataLoader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = DeepFlowMatchingNet(
        in_channels=args.dim,
        model_channels=args.dim,
        out_channels=args.dim,
        num_res_blocks=args.num_res_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()

    for epoch in range(args.epochs):
        loss_epoch = 0
        loss_infoNce_epoch = 0
        for d in dataLoader:
            input_images_feat = d['input_image_feat'].to(device)
            output_images_feat = d['output_image_feat'].to(device)
            output_descriptions = d['output_text_feat'].to(device)
            edit_texts = d['edit_text_feat'].to(device)

            input_images_feat = F.normalize(input_images_feat, dim=-1)
            output_descriptions = F.normalize(output_descriptions, dim=-1)
            edit_texts = F.normalize(edit_texts, dim=-1)
            output_images_feat = F.normalize(output_images_feat, dim=-1)
            loss = 0

            exclude_too_hard_now = True if epoch < args.exclude_too_hard_until_epoch else args.exclude_too_hard
            lambda_info = get_infonce_weight(epoch, args)

            with autocast(enabled=device.type == "cuda"):

                t = torch.rand(input_images_feat.size(0),1,device=device)
                interpolated_features = (1 - t) * edit_texts + t * output_descriptions

                velocity_pred = model(interpolated_features,input_images_feat,t)

                velocity_gt = output_descriptions - edit_texts

                loss_vel = torch.sum((velocity_pred.float()-velocity_gt.float())**2,dim=1).mean()
                loss += loss_vel

                transfered_features = interpolated_features + (1-t)*velocity_pred

                if epoch < args.topk_start_epoch:

                    info_nce_loss = info_nce_topk_hardneg_image_retrieval(
                        transfered_features, output_images_feat,
                        temperature=0.07, normalize=True,
                        k=args.negK,
                        exclude_too_hard=exclude_too_hard_now, eps=args.eps_pre
                    )

                else:
                    info_nce_loss = info_nce_topk_hardneg_image_retrieval(
                        transfered_features, output_images_feat,
                        temperature=0.07, normalize=True,
                        k=args.negK,
                        exclude_too_hard=exclude_too_hard_now, eps=args.eps
                    )
                loss += lambda_info*info_nce_loss

            loss_epoch += loss_vel.item()
            loss_infoNce_epoch += info_nce_loss.item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"[Training] Epoch [{epoch+1}/{args.epochs}], Loss velocity: {loss_epoch:.4f} Loss InfoNCE: {loss_infoNce_epoch:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{epoch+1}.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir,'model.pth'))
    print(f"Trained model saved to {os.path.join(args.save_dir,'model.pth')}")


if __name__ == "__main__":
    main()
