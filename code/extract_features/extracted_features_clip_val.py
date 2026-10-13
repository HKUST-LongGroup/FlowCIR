import os
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLIP_MODEL, CIRR_DIR, CIRCO_DIR, feature_path
from cir_dataset import CIRCODataset, CIRRDataset  # 你已有的 Dataset

import clip  # OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    if any(x is None for x in batch):
        raise ValueError("Dataset returned a missing sample; refusing to save incomplete features")
    return torch.utils.data.dataloader.default_collate(batch)


def collate_relative_batch(batch: List[dict]) -> dict:
    """
    Keep per-sample Python lists/strings as-is instead of letting default_collate
    implicitly transpose nested lists such as group_members / gt_img_ids.
    """
    if any(x is None for x in batch):
        raise ValueError("Dataset returned a missing sample; refusing to save incomplete features")
    if len(batch) == 0:
        return {}

    out = {}
    for key in batch[0].keys():
        values = [x[key] for x in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.utils.data.dataloader.default_collate(values)
        else:
            out[key] = values
    return out

# -------------------------
# collate: test / relative (query)
# -------------------------
def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _parse_clip_name(name: str) -> str:
    """
    支持一些你可能习惯的写法，统一映射到 OpenAI CLIP 的官方名称：
      - ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
    """
    s = name.strip()
    alias = {
        "vitb32": "ViT-B/32",
        "vit-b/32": "ViT-B/32",
        "vitb16": "ViT-B/16",
        "vit-b/16": "ViT-B/16",
        "vitl14": "ViT-L/14",
        "vit-l/14": "ViT-L/14",
        "vitl14@336": "ViT-L/14@336px",
        "vit-l/14@336": "ViT-L/14@336px",
        "vit-l/14@336px": "ViT-L/14@336px",
    }
    key = s.lower().replace(" ", "")
    return alias.get(key, s)  # 如果本来就是官方名，就原样返回


@torch.no_grad()
def main(
    dataset_name:str,
    dataset_path: str,
    save_query_path: str,
    save_gallery_path: str,
    clip_model_name: str = "ViT-B/32",
    joint_fusion: str = "sum",  # "sum" or "mean"
    batch_size: int = 16,
    num_workers: int = 2,
    device: str = "cuda",
):
    dataset_name = dataset_name.lower()
    if dataset_name not in ("cirr", "circo"):
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if Path(save_query_path).resolve() == Path(save_gallery_path).resolve():
        raise ValueError("Query and gallery output paths must differ")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(str(Path(save_query_path).parent), exist_ok=True)
    os.makedirs(str(Path(save_gallery_path).parent), exist_ok=True)

    # ===== 1) Load CLIP =====
    clip_model_name = _parse_clip_name(clip_model_name)
    model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    model.eval()

    # =========================================================
    # Part A: Extract QUERY feats (test / relative) and save
    # =========================================================
    # 你原来传 preprocess=lambda im: im.convert("RGB") 给 dataset
    # 这里我们仍然保证 dataset 返回 PIL.Image，然后在这里用 CLIP preprocess
    # preprocess_pil = lambda im: im.convert("RGB")
    if dataset_name.lower() == "circo":
        query_dataset = CIRCODataset(dataset_path, split="val", mode="relative", preprocess=preprocess)
    elif dataset_name.lower() == "cirr":
        query_dataset = CIRRDataset(dataset_path, split="val", mode="relative", preprocess=preprocess)
    print(f"[Info] Query dataset size (test/relative): {len(query_dataset)}")

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_relative_batch,
    )

    ref_img_feats: List[torch.Tensor] = []
    cap_feats: List[torch.Tensor] = []
    joint_feats: List[torch.Tensor] = []
    query_ids_all: List[str] = []
    gts_img_ids_list = []
    target_names_list = []
    
    reference_names_list = []
    group_members_list = []

    for batch in tqdm(query_loader, desc="Extract query feats (CLIP)"):
        if not batch:
            continue
        
        ref_imgs = batch["reference_image"]     # List[PIL.Image]
        caps = batch["relative_caption"]        # List[str]
        

        # (A) reference_image only -> image features
        img_tensor = ref_imgs.to(device)
        img_feat = model.encode_image(img_tensor)
        # img_feat = _normalize(img_feat)
        batch_n = img_feat.shape[0]
        assert len(caps) == batch_n, f"Caption count mismatch: {len(caps)} vs {batch_n}"

        # (B) caption only -> text features
        input_captions = [
            f"a photo that {caption}" for caption in caps
        ]
        text_tokens = clip.tokenize(input_captions, context_length=77, truncate=True).to(device)
        txt_feat = model.encode_text(text_tokens)
        # txt_feat = _normalize(txt_feat)

        # (C) "image + caption" joint (CLIP 没有 joint encoder，这里做 fusion)
        if joint_fusion == "mean":
            joint = _normalize((img_feat + txt_feat) * 0.5)
        else:  # "sum"
            joint = _normalize(img_feat + txt_feat)

        ref_img_feats.append(img_feat.detach().to("cpu", dtype=torch.float16))
        cap_feats.append(txt_feat.detach().to("cpu", dtype=torch.float16))
        joint_feats.append(joint.detach().to("cpu", dtype=torch.float16))
        
        target_names = batch['target_name']
        assert len(target_names) == batch_n, f"Target name count mismatch: {len(target_names)} vs {batch_n}"
        target_names_list.extend(target_names)
        if dataset_name.lower() == "circo":
            
            qids = batch["query_id"]                # List[str]
            gt_img_ids = batch['gt_img_ids']
        
            assert len(qids) == batch_n, f"Query id count mismatch: {len(qids)} vs {batch_n}"
            assert len(gt_img_ids) == batch_n, f"GT list count mismatch: {len(gt_img_ids)} vs {batch_n}"
            query_ids_all.extend(qids)
            gts_img_ids_list.extend(gt_img_ids)
            reference_names = batch["reference_name"]
            assert len(reference_names) == batch_n, f"Reference name count mismatch: {len(reference_names)} vs {batch_n}"
            reference_names_list.extend(reference_names)
            
        elif dataset_name.lower() == "cirr":
            group_members = batch['group_members']
            reference_names = batch['reference_name']
            assert len(group_members) == batch_n, f"Group member count mismatch: {len(group_members)} vs {batch_n}"
            assert len(reference_names) == batch_n, f"Reference name count mismatch: {len(reference_names)} vs {batch_n}"
            group_members_list.extend(group_members)
            reference_names_list.extend(reference_names)
            

    if dataset_name.lower() == "circo":
        query_payload = {
            "query_id": query_ids_all,
            "reference_image_feat": torch.cat(ref_img_feats, dim=0),
            "relative_caption_feat": torch.cat(cap_feats, dim=0),
            "reference_plus_caption_feat": torch.cat(joint_feats, dim=0),
            "gts_img_ids": gts_img_ids_list,
            "target_names": target_names_list,
            "reference_names": reference_names_list,
        }
    elif dataset_name.lower() == "cirr":
        query_payload = {
            "reference_image_feat": torch.cat(ref_img_feats, dim=0),
            "relative_caption_feat": torch.cat(cap_feats, dim=0),
            "reference_plus_caption_feat": torch.cat(joint_feats, dim=0),
            "target_names": target_names_list,
            "group_members": group_members_list,
            "reference_names": reference_names_list,
        }
    torch.save(query_payload, save_query_path)
    print(f"[OK] Saved query feats to: {save_query_path}")
    print("Query N =", query_payload["reference_image_feat"].shape[0], "D =", query_payload["reference_image_feat"].shape[-1])

    # =========================================================
    # Part B: Extract GALLERY feats (test / classic) and save
    # =========================================================
    if dataset_name.lower() == "circo":
        gallery_dataset = CIRCODataset(dataset_path, split="val", mode="classic", preprocess=preprocess)
    elif dataset_name.lower() == "cirr":
        gallery_dataset = CIRRDataset(dataset_path, split="val", mode="classic", preprocess=preprocess)
    print(f"[Info] Gallery dataset size (test/classic): {len(gallery_dataset)}")

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    index_features: List[torch.Tensor] = []
    index_names: List[str] = []

    for batch in tqdm(gallery_loader, desc="Extract gallery feats (CLIP)"):
        imgs = batch.get('image')          # List[PIL.Image]
        names = batch.get('image_name')   # List[str]
        if imgs is None:
            imgs = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        imgs = imgs.to(device)
        feats = model.encode_image(imgs)
        # feats = _normalize(feats)

        index_features.append(feats.detach().to("cpu", dtype=torch.float16))
        index_names.extend(names)

    gallery_payload = {
        "index_names": index_names,                      # List[str]
        "index_features": torch.cat(index_features, 0),  # Tensor [N, D]
    }
    torch.save(gallery_payload, save_gallery_path)
    print(f"[OK] Saved gallery feats to: {save_gallery_path}")
    print("Gallery N =", len(index_names), "D =", gallery_payload["index_features"].shape[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["circo", "cirr"],
                        help="Which dataset to extract: circo | cirr")
    parser.add_argument("--save_query_path", type=str)
    parser.add_argument("--save_gallery_path", type=str)
    parser.add_argument("--clip_model", type=str, default=CLIP_MODEL,
                        help="CLIP backbone: ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px (or aliases vitb32/vitl14...)")
    parser.add_argument("--joint_fusion", type=str, default="sum", choices=["sum", "mean"],
                        help="How to form reference_plus_caption_feat for CLIP.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.dataset_path = args.dataset_path or str(CIRR_DIR if args.dataset_name == 'cirr' else CIRCO_DIR)
    args.save_query_path = args.save_query_path or str(feature_path(args.dataset_name, 'val', 'query', args.clip_model))
    args.save_gallery_path = args.save_gallery_path or str(feature_path(args.dataset_name, 'val', 'gallery', args.clip_model))

    main(
        dataset_name = args.dataset_name,
        dataset_path=args.dataset_path,
        save_query_path=args.save_query_path,
        save_gallery_path=args.save_gallery_path,
        clip_model_name=args.clip_model,
        joint_fusion=args.joint_fusion,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
