"""CPU regressions; CLIP is stubbed only for serialization/alignment checks."""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train import info_nce_topk_hardneg_image_retrieval
from hq_dataset.hqEdit_dataset_feat_inverse import HQEditFeatShardDataset
from config import feature_path


class FakeEncoder:
    def eval(self):
        return self

    def encode_image(self, images):
        return images.float()

    def encode_text(self, tokens):
        return tokens.float()


class TinyCIR(torch.utils.data.Dataset):
    def __init__(self, path, split, mode, preprocess):
        self.split, self.mode = split, mode

    def __len__(self):
        return 3

    def __getitem__(self, index):
        image = torch.eye(3)[index]
        if self.mode == 'classic':
            return {'image': image, 'image_name': str(index)}
        result = {'reference_image': image, 'reference_name': str(index),
                  'relative_caption': f'caption {index}', 'query_id': str(index),
                  'group_members': [str(index), str((index + 1) % 3)], 'pair_id': index}
        if self.split == 'val':
            result.update(target_name=str(index), gt_img_ids=[str(index), ''])
        return result


class PipelineRegressions(unittest.TestCase):
    def test_contrastive_edges_and_backward(self):
        for batch in (1, 2, 8):
            for exclude in (False, True):
                q = torch.randn(batch, 16, requires_grad=True)
                loss = info_nce_topk_hardneg_image_retrieval(
                    q, torch.randn_like(q), k=512, exclude_too_hard=exclude)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(q.grad).all())

    def test_full_topk_equals_full_inbatch_ce(self):
        q, target = torch.randn(8, 16), torch.randn(8, 16)
        actual = info_nce_topk_hardneg_image_retrieval(q, target, k=7)
        expected = F.cross_entropy(F.normalize(q, dim=-1) @ F.normalize(target, dim=-1).T / .07,
                                   torch.arange(8))
        torch.testing.assert_close(actual, expected)

    def test_split_paths_never_collide(self):
        paths = {feature_path(d, s, k) for d in ('cirr', 'circo')
                 for s in ('val', 'test') for k in ('query', 'gallery')}
        self.assertEqual(len(paths), 8)

    def test_inverse_pairing_and_shard_mismatch(self):
        fields = ['input_image', 'input_text', 'edit_text', 'inverse_edit_text', 'output_image', 'output_text']
        with tempfile.TemporaryDirectory() as tmp:
            for i, field in enumerate(fields):
                torch.save({'tensor': torch.full((2, 4), float(i))}, Path(tmp) / f'{field}_feat_shard0000.pt')
            data = HQEditFeatShardDataset(tmp, verbose=False)
            self.assertEqual(len(data), 4)
            torch.testing.assert_close(data[2]['input_image_feat'], data[0]['output_image_feat'])
            self.assertTrue((data[2]['edit_text_feat'] == 3).all())
            (Path(tmp) / 'edit_text_feat_shard0000.pt').rename(Path(tmp) / 'edit_text_feat_shard0001.pt')
            with self.assertRaisesRegex(ValueError, 'Shard IDs'):
                HQEditFeatShardDataset(tmp, verbose=False)

    def test_cir_serialization_and_metadata_alignment(self):
        fake_clip = types.ModuleType('clip')
        fake_clip.load = lambda *a, **k: (FakeEncoder(), None)
        fake_clip.tokenize = lambda texts, **k: torch.ones(len(texts), 3)
        with patch.dict(sys.modules, {'clip': fake_clip}), tempfile.TemporaryDirectory() as tmp:
            for suffix, split in (('', 'test'), ('_val', 'val')):
                module = importlib.import_module('extract_features.extracted_features_clip' + suffix)
                with patch.object(module, 'clip', fake_clip), patch.object(module, 'CIRRDataset', TinyCIR), patch.object(module, 'CIRCODataset', TinyCIR):
                    for dataset in ('cirr', 'circo'):
                        query, gallery = Path(tmp) / f'{dataset}_{split}_q.pt', Path(tmp) / f'{dataset}_{split}_g.pt'
                        module.main(dataset, tmp, str(query), str(gallery), batch_size=2, num_workers=0, device='cpu')
                        saved = torch.load(query, map_location='cpu')
                        self.assertEqual(saved['reference_image_feat'].shape, (3, 3))
                        self.assertEqual(saved['reference_names'], ['0', '1', '2'])
                        if dataset == 'cirr':
                            self.assertEqual(saved['group_members'], [['0', '1'], ['1', '2'], ['2', '0']])
                        else:
                            self.assertEqual(saved['query_id'], ['0', '1', '2'])
                        if dataset == 'cirr' and split == 'test':
                            self.assertEqual(saved['pair_id'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
