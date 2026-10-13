from typing import Dict, Any, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path



def load_index_from_pt(
    feat_pt_path: str,
  
):
    """
    读取保存好的 .pt，返回：
      - index_features: Tensor[N, D]  (float32 by default)
      - index_names:    LongTensor[N] (每个样本对应的 name_id)
      - id2name: Dict[int,str]        (name_id -> name)
      - name2id: Dict[str,int]        (name -> name_id)
    """
    feat_pt_path = str(Path(feat_pt_path))
    data = torch.load(feat_pt_path, map_location="cpu")

    
    return data

class CIRCOExtractedFeatDataset(Dataset):
    """
    读取你保存的 .pt
    返回：
      - query_id: str
      - reference_name: str
      - ref_img_feat: FloatTensor[D]
      - cap_feat: FloatTensor[D]
      - joint_feat: FloatTensor[D]
    """
    def __init__(self, feat_pt_path: str, dtype: torch.dtype = torch.float32, val=False):
        feat_pt_path = str(Path(feat_pt_path))
        data = torch.load(feat_pt_path, map_location="cpu")
        self.val = val

        self.query_ids: List[str] = data["query_id"]
        # self.reference_names: List[str] = data["reference_name"]   # ✅新增

        self.ref_img_feat: torch.Tensor = data["reference_image_feat"]
        self.cap_feat: torch.Tensor = data["relative_caption_feat"]
        self.joint_feat: torch.Tensor = data["reference_plus_caption_feat"]
        
        self.relative_caption_feat = data["relative_caption_feat"]
        
        if self.val:
            assert "target_names" in data
            assert "gts_img_ids" in data
            self.target_names = data["target_names"]
            self.gts_img_ids = data["gts_img_ids"]

        n = len(self.query_ids)
        # assert len(self.reference_names) == n                       # ✅新增
        assert self.ref_img_feat.shape[0] == n
        assert self.cap_feat.shape[0] == n
        assert self.joint_feat.shape[0] == n

        self.ref_img_feat = self.ref_img_feat.to(dtype)
        self.cap_feat = self.cap_feat.to(dtype)
        self.joint_feat = self.joint_feat.to(dtype)

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.val:
            return {
                "query_id": self.query_ids[idx],                 # str
                # "reference_name": self.reference_names[idx],     # ✅新增 str
                "ref_img_feat": self.ref_img_feat[idx],          # Tensor[D]
                "cap_feat": self.cap_feat[idx],                  # Tensor[D]
                "joint_feat": self.joint_feat[idx],              # Tensor[D]
            }   
        return {
            "query_id": self.query_ids[idx],                 # str
            "ref_img_feat": self.ref_img_feat[idx],          # Tensor[D]
            "cap_feat": self.cap_feat[idx],                  # Tensor[D]
            "joint_feat": self.joint_feat[idx],              # Tensor[D]
            "target_names": self.target_names[idx],          # str
            "gt_img_ids": self.gts_img_ids[idx],            # str
        }


def collate_feat_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch 输出格式：
      - query_id: List[str]
      - reference_name: List[str]
      - ref_img_feat: Tensor[B, D]
      - cap_feat: Tensor[B, D]
      - joint_feat: Tensor[B, D]
    """
    return {
        "query_id": [x["query_id"] for x in batch],
        # "reference_name": [x["reference_name"] for x in batch],  # ✅新增
        "ref_img_feat": torch.stack([x["ref_img_feat"] for x in batch], dim=0),
        "cap_feat": torch.stack([x["cap_feat"] for x in batch], dim=0),
        "joint_feat": torch.stack([x["joint_feat"] for x in batch], dim=0),
    }


if __name__ == "__main__":
    feat_path = "./saved_feats/circo_test_relative_vlm2vec2b.pt"

    ds = CIRCOExtractedFeatDataset(feat_path, dtype=torch.float32)
    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_feat_batch,
    )

    batch = next(iter(dl))
    print("batch keys:", batch.keys())
    print("query_id:", type(batch["query_id"]), len(batch["query_id"]))
    print("reference_name:", type(batch["reference_name"]), len(batch["reference_name"]))
    print("ref_img_feat:", batch["ref_img_feat"].shape, batch["ref_img_feat"].dtype)
    print("cap_feat:", batch["cap_feat"].shape, batch["cap_feat"].dtype)
    print("joint_feat:", batch["joint_feat"].shape, batch["joint_feat"].dtype)
