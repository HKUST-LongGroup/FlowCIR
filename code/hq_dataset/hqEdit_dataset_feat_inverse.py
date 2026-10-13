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
    一次性把所有 shard 的特征读入并拼接成大 Tensor，然后再做“正向 + 反向”扩充（数据量翻倍）：

    正向（原始）：
      input_*  = input_*
      edit     = edit_text_feat
      output_* = output_*

    反向（新增，表示 output -> input）：
      input_*  = output_*
      edit     = inverse_edit_text_feat
      output_* = input_*

    最终 self.* 都是 [2N, D]（数据量 double），__getitem__ 返回字段格式不变。
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
        inv_edit_files = sorted(glob.glob(os.path.join(feat_dir, "inverse_edit_text_feat_shard*.pt")))
        outi_files   = sorted(glob.glob(os.path.join(feat_dir, "output_image_feat_shard*.pt")))
        outt_files   = sorted(glob.glob(os.path.join(feat_dir, "output_text_feat_shard*.pt")))

        assert len(inputi_files) > 0, f"No input_image_feat_shard*.pt found in {feat_dir}"
        assert len(inv_edit_files) > 0, f"No inverse_edit_text_feat_shard*.pt found in {feat_dir}"

        # 可选：如果你希望强一致，打开下面断言
        print("len(inputi_files), len(inputt_files), len(edit_files), len(inv_edit_files), len(outi_files), len(outt_files) = ",
              len(inputi_files), len(inputt_files), len(edit_files), len(inv_edit_files), len(outi_files), len(outt_files))
        assert len(inputi_files) == len(inputt_files) == len(edit_files) == len(inv_edit_files) == len(outi_files) == len(outt_files), \
            "Shard count mismatch among pt groups."

        groups = (inputi_files, inputt_files, edit_files, inv_edit_files, outi_files, outt_files)
        shard_ids = [[os.path.basename(path).rsplit("_shard", 1)[1] for path in group] for group in groups]
        if any(ids != shard_ids[0] for ids in shard_ids[1:]):
            raise ValueError("Shard IDs do not match across feature groups")
        if shard_limit is not None and shard_limit <= 0:
            raise ValueError("shard_limit must be positive or None")

        if shard_limit is not None:
            inputi_files = inputi_files[:shard_limit]
            inputt_files = inputt_files[:shard_limit]
            edit_files   = edit_files[:shard_limit]
            inv_edit_files = inv_edit_files[:shard_limit]
            outi_files   = outi_files[:shard_limit]
            outt_files   = outt_files[:shard_limit]

        self.num_shards = len(inputi_files)

        # 逐个 shard 读取到 list，然后 cat
        inputi_list: List[torch.Tensor] = []
        inputt_list: List[torch.Tensor] = []
        edit_list:   List[torch.Tensor] = []
        inv_edit_list: List[torch.Tensor] = []
        outi_list:   List[torch.Tensor] = []
        outt_list:   List[torch.Tensor] = []

        total_rows = 0
        emb_dim = None

        for si, (fi_i, fi_t, fe, fie, fo_i, fo_t) in enumerate(
            zip(inputi_files, inputt_files, edit_files, inv_edit_files, outi_files, outt_files)
        ):
            ti  = _load_pt_as_tensor(fi_i)
            tit = _load_pt_as_tensor(fi_t)
            te  = _load_pt_as_tensor(fe)
            tie = _load_pt_as_tensor(fie)   # inverse_edit_text_feat
            toi = _load_pt_as_tensor(fo_i)
            tot = _load_pt_as_tensor(fo_t)

            # 维度一致性检查
            assert ti.ndim == tit.ndim == te.ndim == tie.ndim == toi.ndim == tot.ndim == 2, \
                f"Expect [N, D] tensors, got shapes: {ti.shape}, {tit.shape}, {te.shape}, {tie.shape}, {toi.shape}, {tot.shape}"
            assert ti.shape[0] == tit.shape[0] == te.shape[0] == tie.shape[0] == toi.shape[0] == tot.shape[0], \
                f"Row mismatch in shard {si}"
            if emb_dim is None:
                emb_dim = ti.shape[1]
            assert ti.shape[1] == tit.shape[1] == te.shape[1] == tie.shape[1] == toi.shape[1] == tot.shape[1] == emb_dim, \
                f"Dim mismatch in shard {si}"

            # 先转 dtype（在 CPU 上转，避免 cat 后再转占双份内存峰值）
            if self.dtype is not None:
                ti  = ti.to(dtype=self.dtype)
                tit = tit.to(dtype=self.dtype)
                te  = te.to(dtype=self.dtype)
                tie = tie.to(dtype=self.dtype)
                toi = toi.to(dtype=self.dtype)
                tot = tot.to(dtype=self.dtype)

            inputi_list.append(ti)
            inputt_list.append(tit)
            edit_list.append(te)
            inv_edit_list.append(tie)
            outi_list.append(toi)
            outt_list.append(tot)

            total_rows += ti.shape[0]
            if verbose and (si % 10 == 0 or si == self.num_shards - 1):
                print(f"[load] shard {si+1}/{self.num_shards}, rows={ti.shape[0]}, total_rows={total_rows}")

        # 拼接成大 tensor（默认在 CPU）
        input_image_feat_0  = torch.cat(inputi_list, dim=0).contiguous()
        input_text_feat_0   = torch.cat(inputt_list, dim=0).contiguous()
        edit_text_feat_0    = torch.cat(edit_list,   dim=0).contiguous()
        inverse_edit_text_feat_0 = torch.cat(inv_edit_list, dim=0).contiguous()
        output_image_feat_0 = torch.cat(outi_list,   dim=0).contiguous()
        output_text_feat_0  = torch.cat(outt_list,   dim=0).contiguous()

        # 释放 list 引用，帮助 GC
        del inputi_list, inputt_list, edit_list, inv_edit_list, outi_list, outt_list

        # ========= 关键：做“调转”扩充，数据量翻倍 =========
        # 正向部分（N）
        #   (input_image_0, input_text_0, edit_0, output_image_0, output_text_0)
        # 反向部分（N）
        #   (output_image_0, output_text_0, inverse_edit_0, input_image_0, input_text_0)
        self.input_image_feat  = torch.cat([input_image_feat_0,  output_image_feat_0], dim=0).contiguous()
        self.input_text_feat   = torch.cat([input_text_feat_0,   output_text_feat_0],  dim=0).contiguous()
        self.edit_text_feat    = torch.cat([edit_text_feat_0,    inverse_edit_text_feat_0], dim=0).contiguous()
        self.output_image_feat = torch.cat([output_image_feat_0, input_image_feat_0],  dim=0).contiguous()
        self.output_text_feat  = torch.cat([output_text_feat_0,  input_text_feat_0],   dim=0).contiguous()

        # 如果你不再需要 0 版本，可释放引用（可选）
        del input_image_feat_0, input_text_feat_0, edit_text_feat_0
        del inverse_edit_text_feat_0, output_image_feat_0, output_text_feat_0

        # 按需放到 device（注意：放 GPU 会占显存很大！）
        if self.device is not None:
            self.input_image_feat  = self.input_image_feat.to(self.device, non_blocking=True)
            self.input_text_feat   = self.input_text_feat.to(self.device, non_blocking=True)
            self.edit_text_feat    = self.edit_text_feat.to(self.device, non_blocking=True)
            self.output_image_feat = self.output_image_feat.to(self.device, non_blocking=True)
            self.output_text_feat  = self.output_text_feat.to(self.device, non_blocking=True)

        self.total_len = self.input_image_feat.shape[0]  # = 2N

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
