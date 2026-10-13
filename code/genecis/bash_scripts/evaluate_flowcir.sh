# -------------
# DEFINE EVAL PARAMS
# -------------

# INIT
# -------------
# export PYTHONPATH=$PROJECT_ROOT
model_dir="/path/to/HDD/HQ-Edit/output_dir_resBockls=12_lr0.00005_l14_b4096_ep500_v1_hardneg_elip_clip_l14_negK512_eps0.025_seed21_inv_infonce0.3_warm100_infonce_start_epoch5/model_201.pth"

MODEL='ViT-L/14'
# =================== GeneCIS EVAL ===================

for time in 0.5 0.55 0.6 0.7
do
DATASET='focus_attribute'
CUDA_VISIBLE_DEVICES=2 python /path/to/projects/cir_edit/genecis/eval/evaluate_flowcir.py --dataset $DATASET\
                                                                                --model $MODEL\
                                                                                --fm_model_path $model_dir\
                                                                                --time $time

DATASET='change_attribute'
CUDA_VISIBLE_DEVICES=2 python /path/to/projects/cir_edit/genecis/eval/evaluate_flowcir.py --dataset $DATASET\
                                                                                --model $MODEL\
                                                                                --fm_model_path $model_dir\
                                                                                --time $time

DATASET='focus_object'
CUDA_VISIBLE_DEVICES=2 python /path/to/projects/cir_edit/genecis/eval/evaluate_flowcir.py --dataset $DATASET\
                                                                                --model $MODEL\
                                                                                --fm_model_path $model_dir\
                                                                                --time $time

DATASET='change_object'
CUDA_VISIBLE_DEVICES=2 python /path/to/projects/cir_edit/genecis/eval/evaluate_flowcir.py --dataset $DATASET\
                                                                                --model $MODEL\
                                                                                --fm_model_path $model_dir\
                                                                                --time $time
done