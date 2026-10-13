import os
import glob
import io
from typing import Dict, Any, List

from datasets import load_dataset
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.transforms as T
    HAS_TV = True
except Exception:
    HAS_TV = False


# ========= 1) 读前 N 个 parquet shards，合并成一个 HF dataset =========
def load_hqedit_parquets(data_dir: str, n_parquets: int):
    parquets = sorted(glob.glob(os.path.join(data_dir, "HQ-Edit_*.parquet")))
    assert len(parquets) > 0, f"No HQ-Edit_*.parquet found in {data_dir}"
    n = min(n_parquets, len(parquets))
    picked = parquets[:n]
    print(f"[INFO] Found {len(parquets)} shards, loading first {n}:")
    print("  ", picked[0])
    if n > 1:
        print("  ...")
        print("  ", picked[-1])

    ds = load_dataset("parquet", data_files={"train": picked}, split="train")
    print("[INFO] Loaded rows:", len(ds))
    print("[INFO] Columns:", ds.column_names)
    print("[INFO] Features:", ds.features)
    return ds


# ========= 2) bytes -> PIL.Image 解码 =========
def decode_image_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return img


# ========= 3) PyTorch Dataset 包装：返回你训练会用到的字段 =========
class HQEditTorchDataset(Dataset):
    """
    每个 __getitem__ 返回一个 dict：
      - input_text: str
      - edit: str
      - inverse_edit: str
      - output_text: str
      - input_image: PIL.Image 或 torch.FloatTensor[C,H,W]
      - output_image: PIL.Image 或 torch.FloatTensor[C,H,W]
    """
    def __init__(self, hf_ds, image_to_tensor: bool = True, image_size: int = 224):
        self.ds = hf_ds
        self.image_to_tensor = image_to_tensor and HAS_TV
        if image_to_tensor and not HAS_TV:
            print("[WARN] torchvision not available, will return PIL images instead.")
            self.image_to_tensor = False

        if self.image_to_tensor:
            self.tf = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),  # -> float32, [0,1], shape [C,H,W]
            ])
        else:
            self.tf = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]

        in_img = decode_image_from_bytes(ex["input_image"])
        out_img = decode_image_from_bytes(ex["output_image"])

        if self.tf is not None:
            in_img = self.tf(in_img)
            out_img = self.tf(out_img)

        return {
            "input_text": ex["input"],
            "edit": ex["edit"],
            "inverse_edit": ex["inverse_edit"],
            "output_text": ex["output"],
            "input_image": in_img,
            "output_image": out_img,
        }


# ========= 4) collate_fn：把 batch 组织好（文本 list，图片堆成 tensor） =========
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 文本保持 list[str]
    input_text = [x["input_text"] for x in batch]
    edit = [x["edit"] for x in batch]
    inverse_edit = [x["inverse_edit"] for x in batch]
    output_text = [x["output_text"] for x in batch]

    # 图片：如果是 tensor，就 stack；如果是 PIL，就保留 list
    input_img0 = batch[0]["input_image"]
    if torch.is_tensor(input_img0):
        input_image = torch.stack([x["input_image"] for x in batch], dim=0)   # [B,C,H,W]
        output_image = torch.stack([x["output_image"] for x in batch], dim=0) # [B,C,H,W]
    else:
        input_image = [x["input_image"] for x in batch]   # list[PIL]
        output_image = [x["output_image"] for x in batch] # list[PIL]

    return {
        "input_text": input_text,
        "edit": edit,
        "inverse_edit": inverse_edit,
        "output_text": output_text,
        "input_image": input_image,
        "output_image": output_image,
    }


# ========= 5) Demo：DataLoader 迭代，打印每个 iteration 的 item 结构 =========
if __name__ == "__main__":
    DATA_DIR = "/path/to/HDD/HQ-Edit/data"  # TODO 改成你的目录
    N_PARQUETS = 1                          # TODO 你要加载多少个 shard
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    IMAGE_TO_TENSOR = True                  # True: 输出 torch tensor；False: 输出 PIL
    IMAGE_SIZE = 224

    hf_ds = load_hqedit_parquets(DATA_DIR, n_parquets=N_PARQUETS)
    torch_ds = HQEditTorchDataset(hf_ds, image_to_tensor=IMAGE_TO_TENSOR, image_size=IMAGE_SIZE)
    print(f"Dataset size: {len(torch_ds)} samples")
    loader = DataLoader(
        torch_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 看一个 batch 的结构
    batch = next(iter(loader))
    print("\n=== One batch keys ===")
    print(batch.keys())

    print("\n=== Text fields (list[str]) ===")
    print("input_text[0]:", batch["input_text"][0])
    print("edit[0]:", batch["edit"][0])
    print("output_text[0]:", batch["output_text"][0])
    print("inverse_edit[0]:", batch["inverse_edit"][0])

    print("\n=== Image fields ===")
    if torch.is_tensor(batch["input_image"]):
        print("input_image:", type(batch["input_image"]), batch["input_image"].shape, batch["input_image"].dtype)
        print("output_image:", type(batch["output_image"]), batch["output_image"].shape, batch["output_image"].dtype)
        # -> input_image/output_image: torch.FloatTensor [B,3,H,W]
    else:
        print("input_image: list length =", len(batch["input_image"]), "elem type =", type(batch["input_image"][0]))
        print("output_image: list length =", len(batch["output_image"]), "elem type =", type(batch["output_image"][0]))

    # 你也可以正式迭代
    print("\n=== Iterate a few steps ===")
    for step, b in enumerate(loader):
        # 每个 iteration 的 b 就是一个 dict，结构同上
        if step == 3:
            break
        if torch.is_tensor(b["input_image"]):
            print(f"step {step}: B={len(b['input_text'])}, input_image={b['input_image'].shape}, edit_len={len(b['edit'])}")
        else:
            print(f"step {step}: B={len(b['input_text'])}, input_image=list({len(b['input_image'])})")
