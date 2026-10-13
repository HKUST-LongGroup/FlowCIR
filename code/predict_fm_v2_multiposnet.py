#!/usr/bin/env python3
"""Batch FlowCIR v1 inference with one positive and multiple negative texts.

Input: the query .pth used by predict_fm_v2.py, plus cap_pos_feat [N,D],
cap_neg_feat [N,K,D], and optionally cap_neg_valid [N,K]. Legacy negatives
[N,D]/[N] and per-query lists of [K_i,D] tensors are also accepted.

Alternatively, --posneg_json supplies caption_pos (str) and caption_neg
(str or list[str]); records are aligned by CIRCO id or CIRR pairid, never
by list position. CIRR val uses (reference, target_hard) image pairs.
This mode reuses image/original-caption features and encodes only pos/neg
texts. Its default prefixes match extracted_features_clip_posneg.py.

For each valid negative, construct the SpaceVLM direction from
predict_from_image_neg_multi.py; average directions, normalize, then run
the existing image-conditioned v1 Euler flow. Missing negatives fall back
to the original caption. This script does not train or modify checkpoints.
"""

import argparse
import json
import math
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import CLIP_MODEL
from models.fm_condition import DeepFlowMatchingNet


def _matrix(value, name, shape=None):
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must be a tensor [N,D]")
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name}: expected {tuple(shape)}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _plain(value):
    if torch.is_tensor(value):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def load_query_features(path, dataset, val=False):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("Query file must contain a feature dictionary")
    images = _matrix(data.get("reference_image_feat"), "reference_image_feat")
    captions = _matrix(data.get("relative_caption_feat"), "relative_caption_feat", images.shape)
    n, dim = images.shape
    if n == 0 or dim == 0:
        raise ValueError("Query features must not be empty")
    if (images.norm(dim=-1) == 0).any() or (captions.norm(dim=-1) == 0).any():
        raise ValueError("Reference image and original caption features must be nonzero")

    required = ["query_id"] if dataset == "circo" else ["reference_names", "group_members"]
    if val:
        required += ["target_names"] + (["gts_img_ids"] if dataset == "circo" else [])
    elif dataset == "cirr":
        required += ["pair_id"]
    metadata = {}
    for key in dict.fromkeys(required + ["reference_names", "query_id", "pair_id", "group_members", "target_names", "gts_img_ids"]):
        value = data.get(key)
        if value is None:
            if key in required:
                raise ValueError(f"Missing required metadata: {key}")
            continue
        value = _plain(value)
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise ValueError(f"{key} must contain one entry per query ({n})")
        metadata[key] = list(value)
    if dataset == "circo":
        metadata["query_id"] = [str(x) for x in metadata["query_id"]]
        if len(set(metadata["query_id"])) != n:
            raise ValueError("CIRCO query_id values must be unique")
        metadata.setdefault("target_names", [])
        metadata.setdefault("gts_img_ids", [])
    return data, images, captions, metadata


def load_posneg_features(data, n, dim):
    positives = _matrix(data.get("cap_pos_feat"), "cap_pos_feat", (n, dim))
    negatives = data.get("cap_neg_feat")
    lengths = None
    if isinstance(negatives, (list, tuple)):
        if len(negatives) != n:
            raise ValueError("cap_neg_feat list must contain one [K_i,D] tensor per query")
        rows = [_matrix(row, f"cap_neg_feat[{i}]") for i, row in enumerate(negatives)]
        if any(row.shape[1] != dim for row in rows):
            raise ValueError("Negative feature dimensions do not match image features")
        lengths = torch.tensor([len(row) for row in rows])
        negatives = pad_sequence(rows, batch_first=True)
    elif torch.is_tensor(negatives):
        negatives = negatives.detach().to(device="cpu", dtype=torch.float32)
        if negatives.ndim == 2:
            negatives = negatives.unsqueeze(1)
    else:
        raise ValueError("Missing cap_neg_feat; supply posneg features or --posneg_json")
    if negatives.ndim != 3 or negatives.shape[0] != n or negatives.shape[2] != dim:
        raise ValueError(f"cap_neg_feat must have shape [N,K,{dim}] or [N,{dim}]")
    if not torch.isfinite(negatives).all():
        raise ValueError("cap_neg_feat contains non-finite values")
    mask = data.get("cap_neg_valid")
    if mask is None:
        mask = negatives.norm(dim=-1) > 0
    elif lengths is not None and isinstance(mask, (list, tuple)):
        if len(mask) != n:
            raise ValueError("cap_neg_valid must contain one mask per query")
        masks = [torch.as_tensor(row, dtype=torch.bool) for row in mask]
        if any(row.ndim != 1 or len(row) != int(lengths[i]) for i, row in enumerate(masks)):
            raise ValueError("Each negative mask must match its query's K_i")
        mask = pad_sequence(masks, batch_first=True, padding_value=False)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
        if mask.ndim == 1 and negatives.shape[1] == 1:
            mask = mask.unsqueeze(1)
    if mask.shape != negatives.shape[:2]:
        raise ValueError(f"cap_neg_valid must have shape {tuple(negatives.shape[:2])}")
    if lengths is not None:
        padding = torch.arange(negatives.shape[1])[None, :] >= lengths[:, None]
        if (mask & padding).any():
            raise ValueError("Padding positions cannot be valid negatives")
    if (mask & (negatives.norm(dim=-1) == 0)).any():
        raise ValueError("A negative marked valid has a zero feature vector")
    if (mask.any(dim=1) & (positives.norm(dim=-1) == 0)).any():
        raise ValueError("Queries with valid negatives need a nonzero positive feature")
    return positives, negatives, mask


def read_posneg_annotations(path, metadata, dataset, val):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("--posneg_json must contain a list of annotation records")
    if dataset == "circo":
        feature_key, record_key = "query_id", "id"
    elif "pair_id" in metadata:
        feature_key, record_key = "pair_id", "pairid"
    else:
        feature_key, record_key = "reference_names/target_names", "reference/target_hard"
    pair_alignment = record_key == "reference/target_hard"
    if pair_alignment:
        ids = list(zip(map(str, metadata["reference_names"]), map(str, metadata["target_names"])))
    else:
        ids = [str(x) for x in metadata[feature_key]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Cannot uniquely align annotations using {feature_key}")
    by_id = {}
    for record in records:
        keys = ("reference", "target_hard") if pair_alignment else (record_key,)
        if not isinstance(record, dict) or any(key not in record for key in keys):
            raise ValueError(f"Each annotation must contain {record_key}")
        key = (str(record["reference"]), str(record["target_hard"])) if pair_alignment else str(record[record_key])
        if key in by_id:
            raise ValueError(f"Duplicate annotation {record_key}: {key}")
        by_id[key] = record
    positives, negatives = [], []
    for key in ids:
        if key not in by_id:
            raise ValueError(f"No positive/negative annotation for query {key}")
        record = by_id[key]
        if "caption_pos" not in record or "caption_neg" not in record:
            raise ValueError(f"Query {key} needs caption_pos and caption_neg")
        positive, negative = record["caption_pos"], record["caption_neg"]
        negative = [negative] if isinstance(negative, str) or negative is None else negative
        if not isinstance(negative, list) or any(x is not None and not isinstance(x, str) for x in negative):
            raise ValueError(f"Query {key}: caption_neg must be a string or list of strings")
        negative = [x.strip() for x in negative if x is not None and x.strip() and x.strip().lower() != "something"]
        if not isinstance(positive, str) or (negative and not positive.strip()):
            raise ValueError(f"Query {key}: caption_pos must be one nonempty positive text when negatives exist")
        positives.append(positive.strip())
        negatives.append(negative)
    return positives, negatives


@torch.inference_mode()
def encode_annotations(positives, negatives, args, dim, device):
    texts, destinations = [], []
    for i, (positive, row) in enumerate(zip(positives, negatives)):
        if row:
            texts.append(args.pos_prefix + positive)
            destinations.append((i, -1))
            for k, negative in enumerate(row):
                texts.append(args.neg_prefix + negative)
                destinations.append((i, k))
    n, width = len(positives), max(map(len, negatives), default=0)
    pos = torch.zeros(n, dim)
    neg = torch.zeros(n, width, dim)
    mask = torch.zeros(n, width, dtype=torch.bool)
    if not texts:
        return pos, neg, mask
    # Optional: precomputed features and all-fallback queries need no CLIP install.
    import clip

    model, _ = clip.load(args.clip_model_name, device=device, jit=False)
    model.eval()
    for start in tqdm(range(0, len(texts), args.text_batch_size), desc="Encode pos/neg texts"):
        tokens = clip.tokenize(texts[start:start + args.text_batch_size], truncate=True).to(device)
        encoded = F.normalize(model.encode_text(tokens).float(), dim=-1).cpu()
        if encoded.ndim != 2 or encoded.shape[1] != dim:
            raise ValueError("CLIP text dimension does not match query features; use the same backbone")
        if not torch.isfinite(encoded).all() or (encoded.norm(dim=-1) == 0).any():
            raise ValueError("CLIP returned invalid text features")
        for feature, (i, k) in zip(encoded, destinations[start:start + args.text_batch_size]):
            if k == -1:
                pos[i] = feature
            else:
                neg[i, k], mask[i, k] = feature, True
    return pos, neg, mask


@torch.inference_mode()
def spacevlm_direction(positive, negative, t=0.92):
    """The single-negative closed form used by predict_from_image_neg_multi.py."""
    positive = F.normalize(positive.float(), dim=-1)
    negative = F.normalize(negative.float(), dim=-1)
    alpha = math.acos(t)
    theta = torch.acos((positive * negative).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
    denominator = theta.sin().clamp_min(1e-6)
    c1 = torch.sin(theta / 2 + alpha) / denominator
    c2 = torch.sin(alpha - theta / 2) / denominator
    direction = positive * c1[:, None] - negative * c2[:, None]
    small = theta < 1e-3
    direction[small] = (positive - negative)[small]
    return F.normalize(direction, dim=-1)


@torch.inference_mode()
def construct_start(positive, negatives, valid, fallback, t=0.92):
    """Masked mean of per-negative directions; rows with no negatives use fallback."""
    summed = torch.zeros_like(fallback, dtype=torch.float32)
    for k in range(negatives.shape[1]):
        selected = valid[:, k]
        if selected.any():
            summed[selected] += spacevlm_direction(positive[selected], negatives[selected, k], t)
    count = valid.sum(-1, keepdim=True)
    averaged = summed / count.clamp_min(1)
    usable = (count[:, 0] > 0) & (averaged.norm(dim=-1) > 1e-6)
    result = fallback.clone()
    result[usable] = F.normalize(averaged[usable], dim=-1)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Query feature .pth file")
    parser.add_argument("--model_dir", required=True, help="v1 model state_dict .pth")
    parser.add_argument("--save_dir", required=True, help="Output prediction .pth file")
    parser.add_argument("--dataset", required=True, type=str.lower, choices=["circo", "cirr"])
    parser.add_argument("--val", action="store_true", help="Require validation ground-truth metadata")
    parser.add_argument("--model_version", choices=["v1_neg", "v1"], default="v1_neg",
                        help="v1_neg: multi-negative start; v1: original-caption baseline (same network)")
    parser.add_argument("--dim", type=int, default=None, help="Feature dimension; inferred from input if omitted")
    parser.add_argument("--num_res_blocks", type=int, default=12)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--end_point", type=float, default=1.0)
    parser.add_argument("--t", "--spacevlm_t", dest="t", type=float, default=0.92)
    parser.add_argument("--spacevlm_multi_mode", choices=["mean", "min"], default="mean",
                        help="Mean of per-negative directions; min is a compatibility alias for mean, not score-min")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize_inputs", dest="normalize_inputs", action="store_true", default=True)
    parser.add_argument("--no_normalize_inputs", dest="normalize_inputs", action="store_false",
                        help="Keep reference/fallback magnitudes for legacy posneg inference")
    parser.add_argument("--posneg_json", help="Optional annotations with caption_pos and caption_neg string/list")
    parser.add_argument("--clip_model_name", default=CLIP_MODEL, help="Used only with --posneg_json; match feature backbone")
    parser.add_argument("--pos_prefix", default="a photo that ")
    parser.add_argument("--neg_prefix", default="a photo that ")
    parser.add_argument("--text_batch_size", type=int, default=256)
    return parser


@torch.inference_mode()
def run(args):
    if args.steps <= 0 or args.batch_size <= 0 or args.text_batch_size <= 0 or args.num_res_blocks <= 0:
        raise ValueError("steps, batch sizes and num_res_blocks must be positive")
    if not math.isfinite(args.end_point) or not 0 <= args.end_point <= 1:
        raise ValueError("end_point must be between 0 and 1")
    if not math.isfinite(args.t) or not -1 <= args.t <= 1:
        raise ValueError("t must be between -1 and 1")
    if args.posneg_json and args.model_version != "v1_neg":
        raise ValueError("--posneg_json requires --model_version v1_neg")
    destination = Path(args.save_dir).expanduser().resolve()
    if destination in {Path(args.data_dir).expanduser().resolve(), Path(args.model_dir).expanduser().resolve()}:
        raise ValueError("Output must not overwrite query features or checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment")
    data, images, captions, metadata = load_query_features(args.data_dir, args.dataset, args.val)
    n, dim = images.shape
    if args.dim is not None and args.dim != dim:
        raise ValueError(f"--dim {args.dim} does not match feature dimension {dim}")
    if args.spacevlm_multi_mode == "min":
        warnings.warn("min uses mean direction aggregation, as in predict_from_image_neg_multi.py; no score-min is performed")
    if args.model_version == "v1_neg":
        if args.posneg_json:
            positive_texts, negative_texts = read_posneg_annotations(args.posneg_json, metadata, args.dataset, args.val)
            positives, negatives, valid = encode_annotations(positive_texts, negative_texts, args, dim, device)
        else:
            positives, negatives, valid = load_posneg_features(data, n, dim)
        print(f"[Input] queries={n}, dim={dim}, max_negatives={negatives.shape[1]}, "
              f"valid_negatives={int(valid.sum())}, fallback_queries={int((~valid.any(-1)).sum())}")
    model = DeepFlowMatchingNet(in_channels=dim, model_channels=dim, out_channels=dim,
                               num_res_blocks=args.num_res_blocks)
    model.load_state_dict(torch.load(args.model_dir, map_location="cpu", weights_only=True))
    model.to(device).eval()
    times = torch.linspace(0, args.end_point, args.steps + 1, device=device)
    predictions = []
    for start in tqdm(range(0, n, args.batch_size), desc="Predict multi-negative queries"):
        end = min(start + args.batch_size, n)
        ref = images[start:end].to(device)
        original = captions[start:end].to(device)
        if args.normalize_inputs:
            ref, original = F.normalize(ref, dim=-1), F.normalize(original, dim=-1)
        x = original
        if args.model_version == "v1_neg":
            x = construct_start(positives[start:end].to(device), negatives[start:end].to(device),
                                valid[start:end].to(device), original, args.t)
        for i in range(args.steps):
            time = times[i].expand(end - start, 1)
            x = x + model(x, ref, time) * (times[i + 1] - times[i])
        if not torch.isfinite(x).all() or (x.norm(dim=-1) == 0).any():
            raise ValueError(f"Flow produced invalid embeddings for query batch starting at {start}")
        predictions.append(F.normalize(x, dim=-1).cpu().to(torch.float16))
    result = dict(metadata)
    result.update(edited_ref_feats=torch.cat(predictions), relative_caption_feat=captions,
                  relative_img_feat=images.to(torch.float16))
    result["prediction_config"] = dict(dataset=args.dataset, val=args.val, model_version=args.model_version,
                                     dim=dim, num_res_blocks=args.num_res_blocks,
                                     steps=args.steps, end_point=args.end_point, t=args.t,
                                     aggregation="mean", normalize_inputs=args.normalize_inputs,
                                     posneg_json=args.posneg_json, clip_model_name=args.clip_model_name,
                                     pos_prefix=args.pos_prefix, neg_prefix=args.neg_prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    print(f"[OK] Saved {n} query predictions ({dim} dimensions) to {destination}")
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
