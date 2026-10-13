# extract_hqedit_clip_base_feats_4pt.py
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HQEDIT_DIR, HQEDIT_FEATURE_DIR, CLIP_MODEL
import glob
import io
import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset

import clip  # OpenAI CLIP

import argparse


# -----------------------------
# 1) HQ-Edit: bytes -> PIL
# -----------------------------
def bytes_to_pil(b: bytes) -> Image.Image:
    if isinstance(b, Image.Image):
        return b.convert("RGB")
    if isinstance(b, dict):
        if b.get("bytes") is not None:
            b = b["bytes"]
        else:
            return Image.open(b["path"]).convert("RGB")
    return Image.open(io.BytesIO(b)).convert("RGB")


class HQEditShardDataset(Dataset):
    """
    返回：
      - input_text: str
      - edit: str
      - inverse_edit: str
      - output_text: str
      - input_image: PIL.Image
      - output_image: PIL.Image
    """
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        return {
            "input_text": ex["input"],
            "edit": ex["edit"],
            "inverse_edit": ex["inverse_edit"],
            "output_text": ex["output"],
            "input_image": bytes_to_pil(ex["input_image"]),
            "output_image": bytes_to_pil(ex["output_image"]),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_text": [x["input_text"] for x in batch],
        "edit": [x["edit"] for x in batch],
        "inverse_edit": [x["inverse_edit"] for x in batch],
        "output_text": [x["output_text"] for x in batch],
        "input_image": [x["input_image"] for x in batch],    # list[PIL]
        "output_image": [x["output_image"] for x in batch],  # list[PIL]
    }


# -----------------------------
# 2) CLIP feature extraction
# -----------------------------
@torch.no_grad()
def extract_image_reps_clip(
    model,
    preprocess,
    images: List[Image.Image],
    device: str,
    normalize: bool = True,
) -> torch.Tensor:
    x = torch.stack([preprocess(img) for img in images], dim=0).to(device, non_blocking=True)

    if device.startswith("cuda"):
        # CLIP 通常 fp16 推理OK
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            reps = model.encode_image(x)
    else:
        reps = model.encode_image(x)

    reps = reps.float()
    if normalize:
        reps = reps / (reps.norm(dim=-1, keepdim=True) + 1e-6)
    return reps.cpu()


@torch.no_grad()
def extract_text_reps_clip(
    model,
    texts: List[str],
    device: str,
    normalize: bool = True,
) -> torch.Tensor:
    # truncate=True 防止超长文本炸掉
    tokens = clip.tokenize(texts, truncate=True).to(device, non_blocking=True)

    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            reps = model.encode_text(tokens)
    else:
        reps = model.encode_text(tokens)

    reps = reps.float()
    if normalize:
        reps = reps / (reps.norm(dim=-1, keepdim=True) + 1e-6)
    return reps.cpu()


def save_tensor_pt(tensor: torch.Tensor, path: str, shard_path: str):
    # !!! 存储格式不改 !!!
    obj = {
        "tensor": tensor,              # FloatTensor [N, D] on CPU
        "num_rows": tensor.shape[0],
        "dim": tensor.shape[1],
        "shard_path": shard_path,
    }
    torch.save(obj, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(HQEDIT_FEATURE_DIR), help="输出特征存储目录")
    parser.add_argument("--clip_model_name", type=str, default=CLIP_MODEL, help="model name for CLIP")

    parser.add_argument("--data_dir", default=str(HQEDIT_DIR))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ========== 你需要改的地方 ==========
    data_dir = args.data_dir
    OUT_DIR = args.out_dir  # 输出特征存储目录
    N_PARQUETS = None          # None=全量
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = args.device
    NORMALIZE = True

    # “Base 的”：这里用 ViT-B/16（Base-16）
    CLIP_MODEL_NAME = args.clip_model_name
    # ===================================

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- init CLIP (OpenAI clip only) --------
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE, jit=False)
    model.eval()

    # shard list
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "HQ-Edit_*.parquet")))
    assert len(shard_paths) > 0, f"No HQ-Edit_*.parquet found in {data_dir}"
    if N_PARQUETS is not None:
        shard_paths = shard_paths[:N_PARQUETS]

    # meta
    meta = {
        "backend": "openai_clip",
        "clip_model_name": CLIP_MODEL_NAME,
        "normalize": NORMALIZE,
        "batch_size": BATCH_SIZE,
        "dtype": "fp16_compute_float32_save",
        "shards_processed": len(shard_paths),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # process each shard
    for si, shard_path in enumerate(shard_paths):
        print(f"\n[Shard {si}/{len(shard_paths)-1}] {shard_path}")

        hf_ds = load_dataset("parquet", data_files={"train": shard_path}, split="train")
        ds = HQEditShardDataset(hf_ds)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        in_img_list, out_img_list = [], []
        edit_txt_list, inverse_edit_txt_list = [], []
        out_txt_list = []
        in_txt_list = []

        for step, batch in enumerate(loader):
            # input/output image feats
            in_img = extract_image_reps_clip(
                model, preprocess, batch["input_image"], DEVICE, normalize=NORMALIZE
            )
            out_img = extract_image_reps_clip(
                model, preprocess, batch["output_image"], DEVICE, normalize=NORMALIZE
            )

            # edit text feats + inverse_edit text feats + output text feats
            edit_txt = extract_text_reps_clip(
                model, batch["edit"], DEVICE, normalize=NORMALIZE
            )
            inverse_edit_txt = extract_text_reps_clip(
                model, batch["inverse_edit"], DEVICE, normalize=NORMALIZE
            )
            out_txt = extract_text_reps_clip(
                model, batch["output_text"], DEVICE, normalize=NORMALIZE
            )
            input_text = extract_text_reps_clip(
                model, batch["input_text"], DEVICE, normalize=NORMALIZE
            )

            in_img_list.append(in_img)
            out_img_list.append(out_img)
            edit_txt_list.append(edit_txt)
            inverse_edit_txt_list.append(inverse_edit_txt)
            out_txt_list.append(out_txt)
            in_txt_list.append(input_text)

            if step % 50 == 0:
                print(
                    f"  step={step:05d} | B={len(batch['edit'])} "
                    f"| in_img={in_img.shape} | edit_txt={edit_txt.shape} | inverse_edit_txt={inverse_edit_txt.shape}"
                )

        input_image_feat = torch.cat(in_img_list, dim=0)
        output_image_feat = torch.cat(out_img_list, dim=0)
        edit_text_feat = torch.cat(edit_txt_list, dim=0)
        inverse_edit_text_feat = torch.cat(inverse_edit_txt_list, dim=0)
        output_text_feat = torch.cat(out_txt_list, dim=0)
        input_text_feat = torch.cat(in_txt_list, dim=0)

        # save pt per shard (format unchanged)
        save_tensor_pt(
            input_image_feat,
            os.path.join(OUT_DIR, f"input_image_feat_shard{si:04d}.pt"),
            shard_path,
        )
        save_tensor_pt(
            edit_text_feat,
            os.path.join(OUT_DIR, f"edit_text_feat_shard{si:04d}.pt"),
            shard_path,
        )
        save_tensor_pt(
            inverse_edit_text_feat,
            os.path.join(OUT_DIR, f"inverse_edit_text_feat_shard{si:04d}.pt"),
            shard_path,
        )
        save_tensor_pt(
            output_image_feat,
            os.path.join(OUT_DIR, f"output_image_feat_shard{si:04d}.pt"),
            shard_path,
        )
        save_tensor_pt(
            output_text_feat,
            os.path.join(OUT_DIR, f"output_text_feat_shard{si:04d}.pt"),
            shard_path,
        )
        save_tensor_pt(
            input_text_feat,
            os.path.join(OUT_DIR, f"input_text_feat_shard{si:04d}.pt"),
            shard_path,
        )

        print("[OK] Saved pt files for shard", si)
        print("     input_image_feat        :", input_image_feat.shape)
        print("     edit_text_feat          :", edit_text_feat.shape)
        print("     inverse_edit_text_feat  :", inverse_edit_text_feat.shape)
        print("     output_image_feat       :", output_image_feat.shape)
        print("     output_text_feat        :", output_text_feat.shape)
        print("     input_text_feat         :", input_text_feat.shape)

    print("\nAll done.")


if __name__ == "__main__":
    main()
