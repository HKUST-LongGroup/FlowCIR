import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os, sys
from typing import List
from models.fm_condition import DeepFlowMatchingNet, DeepFlowMatchingNet_v1_2, DeepFlowMatchingNet_v2, DeepFlowMatchingNet_v3
from models.fm import DeepFlowMatchingNet as DeepFlowMatchingNet_v0
from hq_dataset.hqEdit_dataset import load_hqedit_parquets, HQEditTorchDataset
from hq_dataset.hqEdit_dataset_feat import HQEditFeatShardDataset
from circo_extracted import CIRCOExtractedFeatDataset
from cirr_extracted import CIRRExtractedFeatDataset
# from cir_dataset import CIRCODataset, CIRRDataset  # 你已有的 Dataset
import sys
import numpy as np
from pathlib import Path
from config import SAVE_ROOT, CLIP_DIMS, CLIP_MODEL
# from src.arguments import ModelArguments, DataArguments
# from src.model.model import MMEBModel
# from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
# from src.utils.basic_utils import batch_to_device
from PIL import Image

import argparse
import clip
from cir_dataset import CIRRDataset, CIRCODataset
from extract_features.extracted_features_clip_val import _parse_clip_name,collate_fn
from tqdm import tqdm

import torchvision.transforms as TVT
from torchvision.transforms.functional import to_pil_image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Dict, Any, Tuple, Union
import torch.nn.functional as F
# ... 你原来的 import 保持不变

def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float, eps: float = 1e-6) -> torch.Tensor:
    """
    a, b: [B, D]
    alpha in [0, 1]
    return: [B, D]  (unit-norm if inputs are unit-norm)
    """
    # normalize onto the unit sphere
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    # cosine similarity and angle
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)  # [B,1]
    theta = torch.acos(dot)                                               # [B,1]
    sin_theta = torch.sin(theta)                                          # [B,1]

    # for very small angles, slerp is numerically unstable -> use lerp
    use_lerp = sin_theta.abs() < eps

    w_a = torch.sin((1.0 - alpha) * theta) / (sin_theta + eps)
    w_b = torch.sin(alpha * theta) / (sin_theta + eps)
    out = w_a * a + w_b * b

    # fallback lerp for small angles
    lerp_out = F.normalize((1.0 - alpha) * a + alpha * b, dim=-1)
    out = torch.where(use_lerp, lerp_out, out)

    return F.normalize(out, dim=-1)


def clip_tensor_batch_to_pil_list(x: torch.Tensor):
    """
    x: [B,3,H,W] normalized by CLIP preprocess
    return: List[PIL.Image]
    """
    x = x.detach().cpu()
    x = (x * _CLIP_STD + _CLIP_MEAN).clamp(0, 1)
    return [to_pil_image(x[i]) for i in range(x.size(0))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='control the number of training data')
    parser.add_argument("--num_res_blocks", type=int, default=8, help="number of residual blocks in DeepFlowMatchingNet")
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the trained model')
    parser.add_argument('--model_dir', type=str, required=True, help='directory to save the trained model')
    parser.add_argument('--steps', type=int, default=10, help='number of steps for flow matching')
    parser.add_argument('--end_point', type=float, default=1.0, help='step size for flow matching')
    parser.add_argument('--dim', type=int, default=CLIP_DIMS[CLIP_MODEL], help='feature dimension')
    parser.add_argument('--normalize_inputs', action='store_true', default=True, help='Normalize v1 inputs to match HQ-Edit training; re-select validation parameters when enabling')
    parser.add_argument('--no_normalize_inputs', action='store_false', dest='normalize_inputs', help='Reproduce legacy unnormalized v1 inference')
    parser.add_argument('--val', action='store_true', help='whether to run validation')
    parser.add_argument("--model_version", type=str, default="v2", help="version of DeepFlowMatchingNet to use")
    parser.add_argument('--clip_model_name', type=str, default='ViT-L/14', help='CLIP model name')
    parser.add_argument('--dataset', type=str, default='cirr', help='dataset name: cirr or CIRCO')

    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    print("args.val =", args.val)
    print("args.model version ", args.model_version)
    print("args.model_dir ", args.model_dir)
    # preprocess = lambda im: im.convert("RGB")
    if "v1" in args.model_version:
        if "circo" in args.dataset or "CIRCO" in args.data_dir:
            dataset = CIRCOExtractedFeatDataset(args.data_dir,  dtype=torch.float32 ,val = args.val)
        elif "cirr" in args.dataset:
            dataset = CIRRExtractedFeatDataset(args.data_dir,  dtype=torch.float32 ,val = args.val)
        dataLoader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    else:
        clip_model_name = _parse_clip_name(args.clip_model_name)
        clip_model, preprocess = clip.load(clip_model_name, device='cuda', jit=False)
        clip_model.eval()
        if "circo" in args.dataset or "CIRCO" in args.data_dir:
            split = "val" if args.val else "test"
            dataset = CIRCODataset(args.data_dir, split=split, mode="relative", preprocess=preprocess)
        elif "cirr" in args.dataset:
            split = "val" if args.val else "test1"
            dataset = CIRRDataset(args.data_dir, split=split, mode="relative", preprocess=preprocess)
        dataLoader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        blip_name = "Salesforce/blip-image-captioning-base"
        blip_processor = BlipProcessor.from_pretrained(blip_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(blip_name).cuda()
        blip_model.eval()
        _CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        _CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    print("dataset length =", len(dataset))


    if "v1" in args.model_version:
        if args.model_version == "v1.2":
            print("args.model_version  ", args.model_version)
            model = DeepFlowMatchingNet_v1_2(in_channels=args.dim,model_channels=args.dim, out_channels=args.dim,num_res_blocks=args.num_res_blocks)
        elif args.model_version == 'v1.3':
            model = DeepFlowMatchingNet_v0(in_channels=args.dim,model_channels=args.dim, out_channels=args.dim,num_res_blocks=args.num_res_blocks).cuda()
        else:
            model = DeepFlowMatchingNet(in_channels=args.dim,model_channels=args.dim, out_channels=args.dim,num_res_blocks=args.num_res_blocks)
    elif "v2" in args.model_version:
        model = DeepFlowMatchingNet_v2(in_channels=args.dim,model_channels=args.dim, out_channels=args.dim,num_res_blocks=args.num_res_blocks)
    elif "v3" in args.model_version:
        model = DeepFlowMatchingNet_v3(in_channels=args.dim,model_channels=args.dim, out_channels=args.dim,num_res_blocks=args.num_res_blocks)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")


    model.load_state_dict(torch.load(args.model_dir))
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)
    Path(args.save_dir).parent.mkdir(parents=True, exist_ok=True)

    ref_feats: List[torch.Tensor] = []
    ref_img_feats: List[torch.Tensor] = []
    if "CIRCO" in args.data_dir or "circo" in args.dataset:

        query_ids_all: List[str] = []
        target_names_list = []
        gt_imgs_ids_list = []
        if args.val:
            target_names_list = dataset.target_names
            gt_imgs_ids_list = dataset.gts_img_ids
    elif "cirr" in args.data_dir or "cirr" in args.dataset:
        print("Processing CIRR dataset")
        group_members = []
        target_names = []
        reference_names_list = []
        if "v1" in args.model_version:
            group_members_list = dataset.group_members
            reference_names_list = dataset.reference_names
            if args.val and "v1" in args.model_version:
                target_names_list = dataset.target_names
        else:
            target_names_list = []
            group_members_list = []
            reference_names_list = []


    if "v1" in args.model_version:
        for batch in tqdm(dataLoader, desc="Extract query feats"):
            ref_imgs = batch["ref_img_feat"]
            ref_img_feats.append(ref_imgs.to("cpu", dtype=torch.float16))
            caps = batch["cap_feat"]        # List[str]
            if args.normalize_inputs:
                caps = F.normalize(caps, dim=-1)
                ref_imgs = F.normalize(ref_imgs, dim=-1)
            if args.dataset.lower() == "circo":

                qids = batch["query_id"]                # List[str]
                query_ids_all.extend(qids)

            ts =  torch.linspace(0,args.end_point,args.steps+1).unsqueeze(1).to('cuda')
            stepsize = args.end_point / args.steps


            if args.model_version == 'v1.3' or args.model_version == 'v1.4':
                transfer_features = slerp(caps, ref_imgs, alpha = 0.15).to('cuda')
            else:
                transfer_features = caps.to('cuda')
            ref_imgs = ref_imgs.to('cuda')
            for i in range(len(ts)-1):
                # print("ts[i]: ", ts[i])
                if args.model_version == "v1.2":
                    N = transfer_features.size(0)
                    t = ts[i].to(transfer_features.device).float().view(1, 1).expand(N, 1)

                    drift = model(transfer_features,ref_imgs, t)

                    # drift = model(transfer_features,ref_imgs, ts[i])
                elif args.model_version == "v1.3":
                    drift = model(transfer_features, ts[i])
                else:
                    drift = model(transfer_features,ref_imgs, ts[i])
                transfer_features = transfer_features +drift*(ts[i+1]-ts[i])
                # print("transfer_features", transfer_features.shape)
            transfer_features = transfer_features / transfer_features.norm(dim=-1, keepdim=True)
            ref_feats.append(transfer_features.detach().to("cpu", dtype=torch.float16))
    else:
        cap_feats: List[torch.Tensor] = []
        for batch in tqdm(dataLoader, desc="Extract query feats"):
            ref_imgs = batch["reference_image"]     # List[PIL.Image]
            caps = batch["relative_caption"]        # List[str]
            device = 'cuda'
            if torch.is_tensor(ref_imgs):
                img_tensor = ref_imgs.to(device, non_blocking=True)
                pil_images = clip_tensor_batch_to_pil_list(ref_imgs)
            else:
                # assume list of PIL
                pil_images = list(ref_imgs)
                img_tensor = torch.stack([preprocess(im) for im in pil_images], dim=0).to(device, non_blocking=True)

            blip_inputs = blip_processor(images=pil_images, return_tensors="pt").to(device)
            blip_out_ids = blip_model.generate(**blip_inputs, max_new_tokens=30)
            blip_caps = blip_processor.batch_decode(blip_out_ids, skip_special_tokens=True)

            img_feat = clip_model.encode_image(img_tensor).float()

            input_captions = [
                f"a photo that {caption}" for caption in caps
            ]
            text_tokens = clip.tokenize(input_captions, context_length=77).to(device)
            edit_text_feat = clip_model.encode_text(text_tokens).float()

            blip_tokens = clip.tokenize(blip_caps, truncate=True).to(device, non_blocking=True)
            img_caption_feat = clip_model.encode_text(blip_tokens).float()

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            edit_text_feat = edit_text_feat / edit_text_feat.norm(dim=-1, keepdim=True)
            img_caption_feat = img_caption_feat / img_caption_feat.norm(dim=-1, keepdim=True)
            cap_feats.append(img_caption_feat.detach().to("cpu", dtype=torch.float16))
            if "CIRCO" in args.data_dir or "circo" in args.dataset:
                qids = batch.get("query_id", None)
                if qids is None:
                    raise KeyError("batch does not contain key 'query_id' for CIRCO")
                query_ids_all.extend(list(qids))
            elif "cirr" in args.data_dir or "cirr" in args.dataset:
                group_members = batch['group_members']
                reference_names = batch['reference_name']
                group_members = np.array(group_members).T.tolist()
                group_members_list.extend(group_members)
                reference_names_list.extend(reference_names)
            target_names = batch['target_name']
            target_names_list.extend(target_names)
            ts = torch.linspace(0, args.end_point, args.steps + 1, device=device)

            if args.model_version == 'v2' or args.model_version == 'v3':
                transfer_features = img_caption_feat
                for i in range(args.steps):
                    dt = (ts[i + 1] - ts[i]).view(1, 1)  # scalar
                    t = ts[i].view(1, 1).expand(transfer_features.size(0), 1).float()

                    drift = model(transfer_features, edit_text_feat, img_feat, t)
                    transfer_features = transfer_features + drift * dt
            else:
                transfer_features = edit_text_feat
                # print("before ode:", transfer_features.shape)
                for i in range(args.steps):
                    dt = (ts[i + 1] - ts[i]).view(1, 1)  # scalar
                    t = ts[i].view(1, 1).expand(transfer_features.size(0), 1).float()

                    drift = model(transfer_features, img_caption_feat, img_feat, t)
                    # print("drift shape: ",drift.shape)
                    transfer_features = transfer_features + drift * dt
                #     print("step", i, "drift:", drift.shape, "transfer:", transfer_features.shape)
                # print("after ode:", transfer_features.shape)



            transfer_features = transfer_features / transfer_features.norm(dim=-1, keepdim=True)
            ref_feats.append(transfer_features.detach().to("cpu", dtype=torch.float16))
            # print("ref_feats shape ",ref_feats.shape)



        # if args.val:
        #     target_names_list.extend(batch['target_names'])
        #     gt_imgs_ids_list.extend(batch['gt_img_ids'])
        #     print("target names ",batch['target_names'])
        #     print("gt images", batch['gt_img_ids'])



        # ref_feats.append(transfer_features.detach().to("cpu", dtype=torch.float16))

    if "CIRCO" in args.data_dir or "circo" in args.dataset:
        if args.val:
            query_payload = {
                "query_id": query_ids_all,
                "edited_ref_feats": torch.cat(ref_feats, dim=0),
                "target_names": target_names_list,
                "gts_img_ids": gt_imgs_ids_list,
                "relative_caption_feat": dataset.cap_feat,
                "relative_img_feat": torch.cat(ref_img_feats, dim=0),
            }
        else:
            query_payload = {
                "query_id": query_ids_all,
                "edited_ref_feats": torch.cat(ref_feats, dim=0),
                "target_names": target_names_list,
                "gts_img_ids": gt_imgs_ids_list,
                "relative_caption_feat": dataset.cap_feat,
                "relative_img_feat": torch.cat(ref_img_feats, dim=0),
            }
    elif "cirr" in args.data_dir or "cirr" in args.dataset:
        if args.val:
            query_payload = {
            "target_names": target_names_list,
            "edited_ref_feats": torch.cat(ref_feats, dim=0),
            "group_members": group_members_list,
            "reference_names": reference_names_list,
            "relative_caption_feat": dataset.relative_caption_feats,
            "relative_img_feat": torch.cat(ref_img_feats, dim=0),
            }
        else:
            query_payload = {
                # "target_names": target_names,
                "edited_ref_feats": torch.cat(ref_feats, dim=0),
                "group_members": group_members_list,
                "reference_names": reference_names_list,
                "relative_caption_feat": dataset.relative_caption_feats,
                "pair_id": dataset.pair_id,
                "relative_img_feat": torch.cat(ref_img_feats, dim=0),
            }
    torch.save(query_payload, args.save_dir)
    print(f"[OK] Saved query feats to: {args.save_dir}")
    print("Query N =", len(ref_feats), "D =", query_payload["edited_ref_feats"].shape[-1])

