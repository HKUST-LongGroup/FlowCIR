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
from cir_dataset_posneg import (
    CIRCODataset_posneg as CIRCODataset,
    CIRRDataset_neg as CIRRDataset,
    FashionIQDataset,
)
import clip
# import open_clip
# from ZS-CIR.data.fiq_dataset import FashionIQDataset  # OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# -------------------------
# collate: test / relative (query)
# -------------------------
def collate_circo_test_relative(batch: List[dict]) -> dict:
    return {
        "reference_images": [x["reference_image"] for x in batch],
        "relative_captions": [x["relative_caption"] for x in batch],
        "query_ids": [x["query_id"] for x in batch],
    }


# -------------------------
# collate: test / classic (gallery)
# -------------------------
def collate_circo_test_classic(batch: List[dict]) -> dict:
    return {
        "images": [x["image"] for x in batch],
        "image_names": [x["image_name"] for x in batch],
    }


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
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(str(Path(save_query_path).parent), exist_ok=True)
    os.makedirs(str(Path(save_gallery_path).parent), exist_ok=True)

    # ===== 1) Load CLIP =====
    if clip_model_name != "vitg14":
        clip_model_name = _parse_clip_name(clip_model_name)
        model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    else:
        def load_clip_vit_g(device="cuda", pretrained="laion2b_s12b_b42k"):
            """
            ViT-G/14 in OpenCLIP.
            Common pretrained choices:
            - "laion2b_s12b_b42k"  (very common)
            - "laion2b_s34b_b88k"  (also seen)
            """
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-g-14",
                pretrained=pretrained,
                device=device,
            )
            model.eval()
            return model, preprocess
        model, preprocess = load_clip_vit_g(device=device, pretrained="laion2b_s12b_b42k")
        
    
    model.eval()

    # =========================================================
    # Part A: Extract QUERY feats (test / relative) and save
    # =========================================================
    if dataset_name.lower() == "circo":
        query_dataset = CIRCODataset(dataset_path, split="test", mode="relative", preprocess=preprocess)
    elif dataset_name.lower() == "cirr":
        query_dataset = CIRRDataset(dataset_path, split="test1", mode="relative", preprocess=preprocess)
    elif dataset_name.lower() == "fashionir":
        query_dataset = FashionIQDataset(dataset_path, split="test", mode="relative", preprocess=preprocess)
    print(f"[Info] Query dataset size (test/relative): {len(query_dataset)}")

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    ref_img_feats: List[torch.Tensor] = []
    cap_feats: List[torch.Tensor] = []
    joint_feats: List[torch.Tensor] = []

    # -------- NEW: cap_pos / cap_neg feats --------
    cap_pos_feats: List[torch.Tensor] = []
    cap_neg_feats: List[torch.Tensor] = []
    cap_neg_valid_all: List[torch.Tensor] = []  # BoolTensor chunks
    # ---------------------------------------------

    query_ids_all: List[str] = []
    group_members_list = []
    reference_names_list = []
    pair_id_list = []

    for batch in tqdm(query_loader, desc="Extract query feats (CLIP)"):
        ref_imgs = batch["reference_image"]     # Tensor [B,3,H,W]
        caps = batch["relative_caption"]        # List[str]

        # -------- NEW: read cap_pos / cap_neg from dataset --------
        cap_pos = batch["cap_pos"]              # List[str]
        cap_neg = batch["cap_neg"]              # List[str]
        # ---------------------------------------------------------

        if dataset_name.lower() == "circo":
            qids = batch["query_id"]                # List[str]
            query_ids_all.extend(qids)
        elif dataset_name.lower() == "cirr":
            group_members = batch['group_members']
            reference_names = batch['reference_name']
            pair_ids = batch['pair_id']
            group_members = np.array(group_members).T.tolist()
            group_members_list.extend(group_members)
            reference_names_list.extend(reference_names)
            pair_id_list.extend(pair_ids)
            # print(f"group members ;{group_members}")
            # print(f"group group_members_list ;{group_members_list}")

        # (A) reference_image only -> image features
        img_tensor = ref_imgs.to(device)
        img_feat = model.encode_image(img_tensor)
        img_feat = _normalize(img_feat)

        # (B) relative_caption -> text features
        input_captions = [f"a photo that {caption}" for caption in caps]
        text_tokens = clip.tokenize(input_captions, context_length=77).to(device)
        txt_feat = model.encode_text(text_tokens)
        txt_feat = _normalize(txt_feat)

        # -------- NEW: cap_pos -> text features --------
        input_captions_pos = [f"a photo that {c}" for c in cap_pos]
        text_tokens_pos = clip.tokenize(input_captions_pos, context_length=77).to(device)
        txt_feat_pos = model.encode_text(text_tokens_pos)
        txt_feat_pos = _normalize(txt_feat_pos)
        cap_pos_feats.append(txt_feat_pos.detach().to("cpu", dtype=torch.float16))
        # ------------------------------------------------

        # -------- NEW: cap_neg -> text features (skip empty) --------
        valid_mask = torch.tensor(
            [
                (c is not None)
                and (str(c).strip() != "")
                and (str(c).strip().lower() != "something")
                for c in cap_neg
            ],
            dtype=torch.bool
        )
        cap_neg_valid_all.append(valid_mask)

        B, D = txt_feat.shape[0], txt_feat.shape[1]
        neg_feat_full = torch.zeros((B, D), device=device, dtype=txt_feat.dtype)

        if valid_mask.any():
            neg_texts = [f"a photo that {cap_neg[i]}" for i, v in enumerate(valid_mask.tolist()) if v]
            neg_tokens = clip.tokenize(neg_texts, context_length=77).to(device)
            neg_feat = model.encode_text(neg_tokens)
            neg_feat = _normalize(neg_feat)

            neg_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1).to(device)
            neg_feat_full[neg_idx] = neg_feat

        cap_neg_feats.append(neg_feat_full.detach().to("cpu", dtype=torch.float16))
        # ------------------------------------------------------------

        # (C) "image + caption" joint (CLIP 没有 joint encoder，这里做 fusion)
        if joint_fusion == "mean":
            joint = _normalize((img_feat + txt_feat) * 0.5)
        else:  # "sum"
            joint = _normalize(img_feat + txt_feat)

        ref_img_feats.append(img_feat.detach().to("cpu", dtype=torch.float16))
        print("img_feat shape ", img_feat.shape)
        cap_feats.append(txt_feat.detach().to("cpu", dtype=torch.float16))
        joint_feats.append(joint.detach().to("cpu", dtype=torch.float16))

    cap_neg_valid = torch.cat(cap_neg_valid_all, dim=0)  # BoolTensor[N]

    if args.dataset_name.lower() == "cirr":
        query_payload = {
            "reference_image_feat": torch.cat(ref_img_feats, dim=0),
            "relative_caption_feat": torch.cat(cap_feats, dim=0),
            "reference_plus_caption_feat": torch.cat(joint_feats, dim=0),

            # -------- NEW --------
            "cap_pos_feat": torch.cat(cap_pos_feats, dim=0),
            "cap_neg_feat": torch.cat(cap_neg_feats, dim=0),
            "cap_neg_valid": cap_neg_valid,
            # ---------------------

            "group_members": group_members_list,
            "reference_names": reference_names_list,
            "pair_id": pair_id_list,
        }
    else:
        query_payload = {
            "query_id": query_ids_all,
            "reference_image_feat": torch.cat(ref_img_feats, dim=0),
            "relative_caption_feat": torch.cat(cap_feats, dim=0),
            "reference_plus_caption_feat": torch.cat(joint_feats, dim=0),

            # -------- NEW --------
            "cap_pos_feat": torch.cat(cap_pos_feats, dim=0),
            "cap_neg_feat": torch.cat(cap_neg_feats, dim=0),
            "cap_neg_valid": cap_neg_valid,
            # ---------------------
        }

    torch.save(query_payload, save_query_path)
    print(f"[OK] Saved query feats to: {save_query_path}")
    print(f"Len {len(group_members_list)}")
    print("Query N =", len(query_payload["reference_image_feat"]), "D =", query_payload["reference_image_feat"].shape[-1])

    # =========================================================
    # Part B: Extract GALLERY feats (test / classic) and save
    # =========================================================
    # if args.dataset_name.lower() == "circo":
    #     gallery_dataset = CIRCODataset(dataset_path, split="test", mode="classic", preprocess=preprocess)
    # elif args.dataset_name.lower() == "cirr":
    #     gallery_dataset = CIRRDataset(dataset_path, split="test1", mode="classic", preprocess=preprocess)
    # else:
    #     gallery_dataset = FashionIQDataset(dataset_path, split="test", mode="classic", preprocess=preprocess)
    # print(f"[Info] Gallery dataset size (test/classic): {len(gallery_dataset)}")

    # gallery_loader = DataLoader(
    #     gallery_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=(device.type == "cuda"),
    #     collate_fn=collate_fn,
    # )

    # index_features: List[torch.Tensor] = []
    # index_names: List[str] = []

    # for batch in tqdm(gallery_loader, desc="Extract gallery feats (CLIP)"):
    #     imgs = batch.get('image')          # List[PIL.Image]
    #     names = batch.get('image_name')   # List[str]
    #     if imgs is None:
    #         imgs = batch.get('reference_image')
    #     if names is None:
    #         names = batch.get('reference_name')

    #     imgs = imgs.to(device)
    #     feats = model.encode_image(imgs)
    #     feats = _normalize(feats)

    #     index_features.append(feats.detach().to("cpu", dtype=torch.float16))
    #     index_names.extend(names)

    # gallery_payload = {
    #     "index_names": index_names,                      # List[str]
    #     "index_features": torch.cat(index_features, 0),  # Tensor [N, D]
    # }
    # torch.save(gallery_payload, save_gallery_path)
    # print(f"[OK] Saved gallery feats to: {save_gallery_path}")
    # print("Gallery N =", len(index_names), "D =", gallery_payload["index_features"].shape[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["circo", "cirr"],
                        help="Which dataset to extract: circo | cirr")
    parser.add_argument("--save_query_path", type=str, required=True)
    parser.add_argument("--save_gallery_path", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP backbone: ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px (or aliases vitb32/vitl14...)")
    parser.add_argument("--joint_fusion", type=str, default="sum", choices=["sum", "mean"],
                        help="How to form reference_plus_caption_feat for CLIP.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        save_query_path=args.save_query_path,
        save_gallery_path=args.save_gallery_path,
        clip_model_name=args.clip_model,
        joint_fusion=args.joint_fusion,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
