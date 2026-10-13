from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


def load_index_from_pt(
    feat_pt_path: str,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str], Dict[str, int]]:
    """
    读取保存好的 gallery .pt（classic 模式保存的那个），返回：
      - index_features: Tensor[N, D]  (dtype 默认 float32)
      - index_names:    LongTensor[N] (每个样本对应的 name_id)
      - id2name: Dict[int,str]        (name_id -> name)
      - name2id: Dict[str,int]        (name -> name_id)

    兼容两种 index_names 存法：
      1) List[str]（常见：你保存的是图片名字符串列表）
      2) LongTensor[N]（若你自己已经做过编码）
    """
    feat_pt_path = str(Path(feat_pt_path))
    data = torch.load(feat_pt_path, map_location="cpu")

    assert "index_features" in data and "index_names" in data, \
        f"Invalid gallery pt: missing keys, got {list(data.keys())}"

    index_features: torch.Tensor = data["index_features"].to(dtype)

    raw_names = data["index_names"]

    # Case A: index_names already LongTensor
    if torch.is_tensor(raw_names):
        index_names = raw_names.to(torch.long).cpu()
        # 如果文件里没提供映射，就无法恢复成字符串
        id2name = data.get("id2name", {})
        name2id = data.get("name2id", {})
        return index_features, index_names, id2name, name2id

    # Case B: index_names is List[str]
    assert isinstance(raw_names, list) and (len(raw_names) == index_features.shape[0]), \
        f"index_names should be List[str] with length N, got type={type(raw_names)} len={len(raw_names) if isinstance(raw_names, list) else 'NA'}"

    names_list: List[str] = raw_names
    uniq = sorted(set(names_list))
    name2id: Dict[str, int] = {n: i for i, n in enumerate(uniq)}
    id2name: Dict[int, str] = {i: n for n, i in name2id.items()}
    index_names = torch.tensor([name2id[n] for n in names_list], dtype=torch.long)

    return index_features, index_names, id2name, name2id


class CIRRExtractedFeatDataset(Dataset):
    """
    读取你保存的 CIRR relative .pt（val split 的 query feats）：
    返回（val/relative）：
      - reference_name: str
      - target_name: str
      - relative_caption: str
      - group_members: List[str]
      - ref_img_feat: FloatTensor[D]
      - cap_feat: FloatTensor[D]
      - joint_feat: FloatTensor[D]
    """
    def __init__(self, feat_pt_path: str, dtype: torch.dtype = torch.float32, val = True):
        feat_pt_path = str(Path(feat_pt_path))
        data = torch.load(feat_pt_path, map_location="cpu")
        self.val = val

        # 必须字段（按你上一个保存脚本的 CIRR 部分）
        for k in [
            "reference_names",
            "group_members",
            "reference_image_feat",
            "relative_caption_feat",
            "reference_plus_caption_feat",
        ]:
            assert k in data, f"Missing key '{k}' in pt. Available keys: {list(data.keys())}"

        self.reference_names: List[str] = data["reference_names"]
        self.target_names: List[str] = data["target_names"] if val else None
        self.group_members: List[List[str]] = data["group_members"]
        # print("len group members ", len(self.group_members), len(self.group_members[0]))
        # print("len reference names ", len(self.reference_names))

        self.ref_img_feat: torch.Tensor = data["reference_image_feat"].to(dtype)
        self.cap_feat: torch.Tensor = data["relative_caption_feat"].to(dtype)
        self.joint_feat: torch.Tensor = data["reference_plus_caption_feat"].to(dtype)
        
        self.relative_caption_feats: torch.Tensor = data["relative_caption_feat"].to(dtype)
        if not val:
            self.pair_id = data["pair_id"]
        else:
            self.pair_id = None

        # n = len(self.reference_names)
        # # assert len(self.target_names) == n
        # assert len(self.group_members) == n
        # assert self.ref_img_feat.shape[0] == n
        # assert self.cap_feat.shape[0] == n
        # assert self.joint_feat.shape[0] == n

    def __len__(self) -> int:
        return len(self.reference_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.val:
            return {
                "reference_name": self.reference_names[idx],      # str
                "target_name": self.target_names[idx],            # str
                # "group_members": self.group_members[idx],         # List[str]
                "ref_img_feat": self.ref_img_feat[idx],           # Tensor[D]
                "cap_feat": self.cap_feat[idx],                   # Tensor[D]
                "joint_feat": self.joint_feat[idx],               # Tensor[D]
            }
        else:
            return {
                "reference_name": self.reference_names[idx],      # str
                # "target_name": self.target_names[idx],            # str
                # "group_members": self.group_members[idx],         # List[str]
                "ref_img_feat": self.ref_img_feat[idx],           # Tensor[D]
                "cap_feat": self.cap_feat[idx],                   # Tensor[D]
                "joint_feat": self.joint_feat[idx],               # Tensor[D]
                "pair_id": self.pair_id[idx],                       # int
            }


def collate_cirr_feat_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader collate:
      - reference_name: List[str]
      - target_name: List[str]
      - relative_caption: List[str]
      - group_members: List[List[str]]
      - ref_img_feat/cap_feat/joint_feat: Tensor[B, D]
    """
    return {
        "reference_name": [x["reference_name"] for x in batch],
        "target_name": [x["target_name"] for x in batch],
        "relative_caption": [x["relative_caption"] for x in batch],
        "group_members": [x["group_members"] for x in batch],
        "ref_img_feat": torch.stack([x["ref_img_feat"] for x in batch], dim=0),
        "cap_feat": torch.stack([x["cap_feat"] for x in batch], dim=0),
        "joint_feat": torch.stack([x["joint_feat"] for x in batch], dim=0),
    }


if __name__ == "__main__":
    # 例子：CIRR val relative 的 query feats
    feat_path = "./saved_feats/cirr_val_relative_clip.pt"

    ds = CIRRExtractedFeatDataset(feat_path, dtype=torch.float32)
    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cirr_feat_batch,
    )

    batch = next(iter(dl))
    print("batch keys:", batch.keys())
    print("reference_name:", type(batch["reference_name"]), len(batch["reference_name"]))
    print("target_name:", type(batch["target_name"]), len(batch["target_name"]))
    print("relative_caption:", type(batch["relative_caption"]), len(batch["relative_caption"]))
    print("group_members:", type(batch["group_members"]), len(batch["group_members"]), "example_len=", len(batch["group_members"][0]))
    print("ref_img_feat:", batch["ref_img_feat"].shape, batch["ref_img_feat"].dtype)
    print("cap_feat:", batch["cap_feat"].shape, batch["cap_feat"].dtype)
    print("joint_feat:", batch["joint_feat"].shape, batch["joint_feat"].dtype)

    # 例子：加载 CIRR gallery feats（classic 保存的那个）
    # gallery_path = "./saved_feats/cirr_val_classic_clip.pt"
    # index_features, index_names, id2name, name2id = load_index_from_pt(gallery_path, dtype=torch.float32)
    # print("index_features:", index_features.shape, index_features.dtype)
    # print("index_names:", index_names.shape, index_names.dtype)
    # print("id2name size:", len(id2name), "name2id size:", len(name2id))
