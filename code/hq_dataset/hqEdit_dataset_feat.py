import os
import glob
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


def _load_pt_as_tensor(fp: str) -> torch.Tensor:
    """
    兼容两种保存格式：
    1) 直接保存 Tensor
    2) 保存 dict，tensor 在 obj["tensor"]
       可能还有 num_rows 等字段
    """
    obj = torch.load(fp, map_location="cpu")
    if isinstance(obj, dict):
        if "tensor" in obj:
            return obj["tensor"]
        for k in ["feat", "feats", "data", "emb", "embedding"]:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        raise KeyError(
            f"Dict in {fp} has no 'tensor' key (and no known fallback keys). "
            f"Keys={list(obj.keys())[:20]}"
        )
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unsupported object type in {fp}: {type(obj)}")


class HQEditFeatShardDataset(Dataset):
    """
    一次性把所有 shard 的 5 类特征读入并拼接成大 Tensor：
      - self.input_image_feat:  [N, D]
      - self.input_text_feat:   [N, D]
      - self.edit_text_feat:    [N, D]
      - self.output_image_feat: [N, D]
      - self.output_text_feat:  [N, D]

    __getitem__(idx) 直接索引，不再 torch.load。
    """
    def __init__(
        self,
        feat_dir: str,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        shard_limit: Optional[int] = None,  # 只用前 N 个 shard（调试用）
        verbose: bool = True,
    ):
        self.feat_dir = feat_dir
        self.dtype = dtype
        self.device = device

        # 找所有 shard 文件（以 input_image 为准）
        inputi_files = sorted(glob.glob(os.path.join(feat_dir, "input_image_feat_shard*.pt")))
        inputt_files = sorted(glob.glob(os.path.join(feat_dir, "input_text_feat_shard*.pt")))
        edit_files   = sorted(glob.glob(os.path.join(feat_dir, "edit_text_feat_shard*.pt")))
        outi_files   = sorted(glob.glob(os.path.join(feat_dir, "output_image_feat_shard*.pt")))
        outt_files   = sorted(glob.glob(os.path.join(feat_dir, "output_text_feat_shard*.pt")))

        assert len(inputi_files) > 0, f"No input_image_feat_shard*.pt found in {feat_dir}"
        # assert len(inputi_files) == len(inputt_files) == len(edit_files) == len(outi_files) , \
        #     "Shard count mismatch among 5 pt groups."

        if shard_limit is not None:
            inputi_files = inputi_files[:shard_limit]
            inputt_files = inputt_files[:shard_limit]
            edit_files   = edit_files[:shard_limit]
            outi_files   = outi_files[:shard_limit]
            outt_files   = outt_files[:shard_limit]

        self.num_shards = len(inputi_files)

        # 逐个 shard 读取到 list，然后 cat
        inputi_list: List[torch.Tensor] = []
        inputt_list: List[torch.Tensor] = []
        edit_list:   List[torch.Tensor] = []
        outi_list:   List[torch.Tensor] = []
        outt_list:   List[torch.Tensor] = []

        total_rows = 0
        emb_dim = None

        for si, (fi_i, fi_t, fe, fo_i, fo_t) in enumerate(
            zip(inputi_files, inputt_files, edit_files, outi_files, outt_files)
        ):
            ti = _load_pt_as_tensor(fi_i)
            tit = _load_pt_as_tensor(fi_t)
            te = _load_pt_as_tensor(fe)
            toi = _load_pt_as_tensor(fo_i)
            tot = _load_pt_as_tensor(fo_t)

            # 维度一致性检查
            assert ti.ndim == tit.ndim == te.ndim == toi.ndim ==  2, \
                f"Expect [N, D] tensors, got shapes: {ti.shape}, {tit.shape}, {te.shape}, {toi.shape}"
            assert ti.shape[0] == tit.shape[0] == te.shape[0] == toi.shape[0] == tot.shape[0], \
                f"Row mismatch in shard {si}"
            if emb_dim is None:
                emb_dim = ti.shape[1]
            assert ti.shape[1] == tit.shape[1] == te.shape[1] == toi.shape[1]  == emb_dim, \
                f"Dim mismatch in shard {si}"

            # 先转 dtype（在 CPU 上转，避免 cat 后再转占双份内存峰值）
            if self.dtype is not None:
                ti = ti.to(dtype=self.dtype)
                tit = tit.to(dtype=self.dtype)
                te = te.to(dtype=self.dtype)
                toi = toi.to(dtype=self.dtype)
                tot = tot.to(dtype=self.dtype)

            inputi_list.append(ti)
            inputt_list.append(tit)
            edit_list.append(te)
            outi_list.append(toi)
            outt_list.append(tot)

            total_rows += ti.shape[0]
            if verbose and (si % 10 == 0 or si == self.num_shards - 1):
                print(f"[load] shard {si+1}/{self.num_shards}, rows={ti.shape[0]}, total_rows={total_rows}")

        # 拼接成大 tensor（默认在 CPU）
        self.input_image_feat  = torch.cat(inputi_list, dim=0).contiguous()
        self.input_text_feat   = torch.cat(inputt_list, dim=0).contiguous()
        self.edit_text_feat    = torch.cat(edit_list,   dim=0).contiguous()
        self.output_image_feat = torch.cat(outi_list,   dim=0).contiguous()
        self.output_text_feat  = torch.cat(outt_list,   dim=0).contiguous()

        # 释放 list 引用，帮助 GC
        del inputi_list, inputt_list, edit_list, outi_list, outt_list

        # 按需放到 device（注意：放 GPU 会占显存很大！）
        if self.device is not None:
            self.input_image_feat  = self.input_image_feat.to(self.device, non_blocking=True)
            self.input_text_feat   = self.input_text_feat.to(self.device, non_blocking=True)
            self.edit_text_feat    = self.edit_text_feat.to(self.device, non_blocking=True)
            self.output_image_feat = self.output_image_feat.to(self.device, non_blocking=True)
            self.output_text_feat  = self.output_text_feat.to(self.device, non_blocking=True)

        self.total_len = self.input_image_feat.shape[0]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_image_feat":  self.input_image_feat[idx],
            "input_text_feat":   self.input_text_feat[idx],
            "edit_text_feat":    self.edit_text_feat[idx],
            "output_image_feat": self.output_image_feat[idx],
            "output_text_feat":  self.output_text_feat[idx],
            "global_idx": idx,
        }


def collate_feat_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_image_feat":  torch.stack([b["input_image_feat"] for b in batch], dim=0),
        "input_text_feat":   torch.stack([b["input_text_feat"] for b in batch], dim=0),
        "edit_text_feat":    torch.stack([b["edit_text_feat"] for b in batch], dim=0),
        "output_image_feat": torch.stack([b["output_image_feat"] for b in batch], dim=0),
        "output_text_feat":  torch.stack([b["output_text_feat"] for b in batch], dim=0),
        "global_idx":        torch.tensor([b["global_idx"] for b in batch], dtype=torch.long),
    }


if __name__ == "__main__":
    FEAT_DIR = "/path/to/save/features_5pt"

    ds = HQEditFeatAllInMemoryDataset(
        feat_dir=FEAT_DIR,
        dtype=torch.float16,   # 如果内存吃紧可以 float16 / bfloat16
        device="cpu",          # 建议先放 CPU；要全放 GPU 就改成 "cuda"
        shard_limit=None,
        verbose=True,
    )

    loader = DataLoader(
        ds,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_feat_batch,
        pin_memory=True,
        drop_last=False,
    )

    batch = next(iter(loader))
    print("input_image_feat:", batch["input_image_feat"].shape, batch["input_image_feat"].dtype)
    print("input_text_feat :", batch["input_text_feat"].shape,  batch["input_text_feat"].dtype)
