# CUDA_VISIBLE_DEVICES=1 python3 generate_submission_vlm2vec.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_vlm2vec_2b_extracted_feats.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_vlm2vec_2b_gallery_feats.pth \
#     --retrieval_type image_text_combined \
#     --submission_name 'image_text_combined' \
#     --dataset "circo" \
#     --exp_name 'image_text_combined'



# CUDA_VISIBLE_DEVICES=1 python3 generate_submission_vlm2vec.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_query_fm_edited_feats_fm_s5.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_vlm2vec_2b_gallery_feats.pth \
#     --retrieval_type fm_edited \
#     --submission_name 'fm_edited_s5' \
#     --dataset "circo" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 generate_submission_vlm2vec.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_query_fm_edited_feats_fm_s5_clipb16.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_clip_vitb16_gallery_feats.pth \
#     --retrieval_type fm_edited \
#     --submission_name 'submissions/fm_edited_clipb16_s5' \
#     --dataset "circo" \
#     --exp_name 'fm_edited_clipb16'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_query_fm_edited_feats_fm_s1_clipb16_val_e5_s10.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitb16_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "circo" \
#     --exp_name 'fm_edited_clipb16'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitb16_query_feats_val.pth  \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitb16_gallery_feats_val.pth \
#     --retrieval_type text_only \
#     --dataset "circo" \
#     --exp_name 'text_only'

# image_only | text_only | image_text_combined | fm_edited_text_combined | fm_edited

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitb16_query_feats_val.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitb16_gallery_feats_val.pth \
#     --retrieval_type image_text_combined \
#     --dataset "cirr" \
#     --exp_name 'image_text_combined'


# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block12_epoch501_infonce01_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block24_epoch201_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block12_epoch2001_infonce01_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block12_epoch4001_infonce01_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitl14_query_feats_val.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type image_only \
#     --dataset "circo" \
#     --exp_name 'image_only'


# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block12_v2_epoch101_infonce00_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block12_v2_epoch201_infonce01_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_query_fm_edited_feats_fm_clipl14_block12_v2_epoch4901_infonce01_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited \
#     --dataset "cirr" \
#     --exp_name 'fm_edited'

CUDA_VISIBLE_DEVICES=1 python3 generate_submission_vlm2vec.py \
    --dataset_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_query_fm_edited_feats_fm_clipl14_block12_v2_epoch101_infonce00_e1_s1.pth \
    --dataset_index_path /path/to/HDD/retrieval_data/data/cirr/extracted_features/CIRR_clip_vitl14_gallery_feats.pth \
    --retrieval_type fm_edited \
    --submission_name 'fm_edited_cirr' \
    --dataset "cirr" \
    --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 generate_submission_vlm2vec.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_query_fm_edited_feats_fm_clipl14_block12_v2_epoch201_infonce00_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitl14_gallery_feats.pth \
#     --retrieval_type fm_edited_text_combined \
#     --submission_name 'fm_edited_text_combined' \
#     --dataset "circo" \
#     --exp_name 'fm_edited'

# CUDA_VISIBLE_DEVICES=1 python3 validate.py \
#     --dataset_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_query_fm_edited_feats_fm_clipl14_block12_v2_epoch201_infonce00_val_e1_s1.pth \
#     --dataset_index_path /path/to/HDD/retrieval_data/CIRCO_extracted_features/CIRCO_clip_vitl14_gallery_feats_val.pth \
#     --retrieval_type fm_edited_text_combined \
#     --dataset "circo" \
#     --exp_name 'fm_edited_text_combined'