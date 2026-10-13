import json
import pickle
from argparse import ArgumentParser
from typing import List, Tuple, Dict


import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm


from cir_dataset import CIRRDataset, CIRCODataset
from circo_extracted import load_index_from_pt
# from encode_with_pseudo_tokens import encode_with_pseudo_tokens
# from phi import Phi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import SUBMISSION_DIR
PROJECT_ROOT = str(SUBMISSION_DIR)


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

@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str,dataset_index_path: str,
                                        submission_name: str, retrieval_type:str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    index_data = load_index_from_pt(dataset_index_path)
    index_features, index_names = index_data['index_features'], index_data['index_names']

    data = load_index_from_pt(dataset_path)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(data, index_features, index_names,
                                 retrieval_type)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)


    # submissions_folder_path.mkdir(exist_ok=True, parents=True)

    submissions_folder_path = PROJECT_ROOT 
    # submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path + '/'+ f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path+'/'+f"subset_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(data, index_features: torch.Tensor,
                             index_names: List[str], retrieval_type: str) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, reference_names, pairs_id, group_members = \
        cirr_generate_test_predictions(data, retrieval_type)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()
    
    predicted_features = predicted_features.to(device)
    predicted_features = F.normalize(predicted_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(data,  retrieval_type:str) -> Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """
    
    print("data keys ",data.keys())
    if retrieval_type == 'image_only':
        predicted_features = data['reference_image_feat']
    elif retrieval_type == 'text_only':
        predicted_features = data['relative_caption_feat']
    elif retrieval_type == 'fm_edited':
        predicted_features = data['edited_ref_feats']
    elif retrieval_type == 'fm_edited_text_combined':
        predicted_features = data['edited_ref_feats'] + data['relative_caption_feat']
    elif retrieval_type == 'fm_edited_img_combined':
        predicted_features = data['edited_ref_feats'] + data['relative_img_feat']
        # query_ids_list = data['query_id']
    elif retrieval_type == 'fm_edited_img_combined_slerp':
        alpha = 0.1
        predicted_features = slerp(data['edited_ref_feats'], data['relative_img_feat'], alpha)
    else:  # 'image_text_combined'
        predicted_features = data['reference_plus_caption_feat']
    
    
    group_members_list = data['group_members']
    reference_names_list = data['reference_names']
    pair_id_list = data['pair_id']


    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str,dataset_index_path: str,
                                        submission_name: str, retrieval_type:str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    
    # Compute the index features
    # classic_test_dataset = CIRCOExtractedFeatDataset(dataset_path, dtype=torch.float32)
    index_data = load_index_from_pt(dataset_index_path)
    print("data",index_data.keys())
    index_features, index_names = index_data['index_features'], index_data['index_names']
    print("index_names   ",index_names.__len__())
    print("index_names   ",set(index_names).__len__())

    # relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess,flow_edit=flow_edit)

    # Get the predictions dict
    data = load_index_from_pt(dataset_path)
    queryid_to_retrieved_images = circo_generate_test_dict(data, index_features,
                                                           index_names,retrieval_type)

    submissions_folder_path = PROJECT_ROOT  
    # submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path + '/' + f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)



def circo_generate_test_predictions(
        data,
        retrieval_type: str,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Generate the test prediction features for the CIRCO dataset
    using plain CLIP image/text features (no pseudo tokens).
    """

    

    predicted_features_list: List[torch.Tensor] = []
    query_ids_list: List[str] = []
    
    if retrieval_type == 'image_only':
        predicted_features = data['reference_image_feat']
        query_ids_list = data['query_id']
    elif retrieval_type == 'text_only':
        predicted_features = data['relative_caption_feat']
        query_ids_list = data['query_id']
    elif retrieval_type == 'fm_edited':
        predicted_features = data['edited_ref_feats']
        query_ids_list = data['query_id']
    elif retrieval_type == 'fm_edited_text_combined':
        predicted_features = data['edited_ref_feats'] + data['relative_caption_feat']
        query_ids_list = data['query_id']
    elif retrieval_type == 'fm_edited_img_combined':
        predicted_features = data['edited_ref_feats'] + data['relative_img_feat']
        query_ids_list = data['query_id']
    else:  # 'image_text_combined'
        predicted_features = data['reference_plus_caption_feat']
        query_ids_list = data['query_id']
    print("query_ids_list[:5]",query_ids_list[:5])

    return predicted_features, query_ids_list

def circo_generate_test_dict(
        data,
        index_features: torch.Tensor,
        index_names: List[str],
        retrieval_type = 'image_only'
) -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset
    using CLIP-based retrieval (no pseudo tokens).
    """

    # Get the predicted features (queries)
    predicted_features, query_ids = circo_generate_test_predictions(
        data,retrieval_type=retrieval_type
    )

    # Normalize the index features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)
    
    predicted_features = predicted_features.float().to(device)
    predicted_features = F.normalize(predicted_features, dim=-1)

    # Compute the similarity: [num_queries, num_index]
    similarity = predicted_features @ index_features.T

    # Top-50 retrieval
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images: Dict[str, List[str]] = {
        query_id: query_sorted_names[:50].tolist()
        for query_id, query_sorted_names in zip(query_ids, sorted_index_names)
    }

    return queryid_to_retrieved_images


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission_name", type=str, required=True, help="Filename of the generated submission file")
    parser.add_argument("--exp_name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'circo'], help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--dataset_index_path", type=str, help="Path to the dataset", required=True)


  
    parser.add_argument("--retrieval_type", type=str,
                        help="image_only | text_only | image_text_combined")


    args = parser.parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)



  
    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path,args.dataset_index_path, args.submission_name, retrieval_type=args.retrieval_type)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path,args.dataset_index_path, args.submission_name, retrieval_type=args.retrieval_type)

    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
