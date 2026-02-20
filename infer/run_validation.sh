# fullset !!
# pretrained_model_name_or_path="/lpai/volumes/ad-vla-vol-ga/chenantong/ckpts/CogVideoX-5b-I2V"
# checkpoint_path="../train/nuscenes-inpainting/2nodes/checkpoint-28000"
# # meta_file_path="/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_val_fullset.json"
# meta_file_path="/lpai/volumes/ad-wm-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_val_selected1k.json"
# # condition_data_root="/lpai/dataset/nuscenes-val-render-mons/0-1-0/nuscenes_vistaval_balanced1k_stride1window3_nonedit"
# # condition_data_root="/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/renderings"
# condition_data_root="/lpai/volumes/ad-wm-vol-ga/chenantong/datasets/NuScenes/renderings_shuffletraj"
# video_data_root="/lpai/dataset/nuscenes-imgs/0-1-0/nuscenes"

#  for gpu in 0 1 2 3 4 5 6 7; do
#    CUDA_VISIBLE_DEVICES=$gpu python run_validation.py \
#      --checkpoint_path $checkpoint_path \
#      --pretrained_model_name_or_path $pretrained_model_name_or_path \
#      --meta_file_path $meta_file_path \
#      --condition_data_root $condition_data_root \
#      --video_data_root $video_data_root \
#      --output_dir validation_results/2nodes_28ksteps/shuffle1k \
#      --height 480 \
#      --width 720 \
#      --max_num_frames 49 \
#      --mixed_precision bf16 \
#      --target_frames 25 \
#      --total_sections 16 \
#      --section $((gpu + 8)) &
#  done

# Evaluate different checkpoints
#steps=(10000 20000 22000 24000 26000 28000)
#for i in "${!steps[@]}"; do
#  checkpoint_step=${steps[$i]}
#  checkpoint_path="../train/nuscenes-inpainting/2nodes/checkpoint-${checkpoint_step}"
#  meta_file_path="/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_val_1k_sampled_10.json"
#  
#  CUDA_VISIBLE_DEVICES=$i python run_validation.py \
#    --checkpoint_path $checkpoint_path \
#    --pretrained_model_name_or_path $pretrained_model_name_or_path \
#    --meta_file_path $meta_file_path \
#    --condition_data_root $condition_data_root \
#    --video_data_root $video_data_root \
#    --output_dir validation_results/subset10/2nodes_${checkpoint_step}steps \
#    --height 480 \
#    --width 720 \
#    --max_num_frames 49 \
#    --mixed_precision bf16 \
#    --target_frames 25 \
#    --total_sections 1 \
#    --section 0 &
#done

# pretrained_model_name_or_path="/lpai/volumes/ad-vla-vol-ga/chenantong/ckpts/CogVideoX-5b-I2V"
# checkpoint_path="../train/nuscenes-inpainting/scratch_nus_branch_nonedit/checkpoint-10000"
# meta_file_path="/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_val_selected1k.json"
# condition_data_root="/lpai/dataset/nuscenes-val-render-mons/0-1-0/nuscenes_vistaval_balanced1k_stride1window3_nonedit"
# video_data_root="/lpai/dataset/nuscenes-imgs/0-1-0/nuscenes"

# for gpu in 0 1 2 3 4 5 6 7; do
#   CUDA_VISIBLE_DEVICES=$gpu python run_validation.py \
#     --checkpoint_path $checkpoint_path \
#     --pretrained_model_name_or_path $pretrained_model_name_or_path \
#     --meta_file_path $meta_file_path \
#     --condition_data_root $condition_data_root \
#     --video_data_root $video_data_root \
#     --output_dir validation_results/scratch_nus_branch_nonedit_10ksteps \
#     --height 480 \
#     --width 720 \
#     --max_num_frames 49 \
#     --mixed_precision bf16 \
#     --target_frames 25 \
#     --total_sections 8 \
#     --section $((gpu)) &
#   done

# custom traj
pretrained_model_name_or_path="THUDM/CogVideoX-5b-I2V"  # or local path to downloaded CogVideoX-5b-I2V
checkpoint_path="../checkpoints/geodrive-branch"
meta_file_path="./custom_traj/custom_traj_meta.json"
condition_data_root="./custom_traj"
video_data_root="./custom_traj"

for gpu in 0; do
  CUDA_VISIBLE_DEVICES=$gpu python run_validation.py \
    --checkpoint_path $checkpoint_path \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --meta_file_path $meta_file_path \
    --condition_data_root $condition_data_root \
    --video_data_root $video_data_root \
    --output_dir validation_results/custom_traj \
    --height 480 \
    --width 720 \
    --conditioning_scale 1.0 \
    --max_num_frames 49 \
    --mixed_precision bf16 \
    --target_frames 49 \
    --total_sections 1 \
    --section 0 &
done

# novel view
# pretrained_model_name_or_path="/lpai/volumes/ad-vla-vol-ga/chenantong/ckpts/CogVideoX-5b-I2V"
# checkpoint_path="../train/nuscenes-inpainting/2nodes/checkpoint-28000"
# # lora_path="../train/nuscenes-inpainting/resume10k_nuswaymo_loraonly/checkpoint-14000"
# meta_file_path="./custom_novelview/custom_traj_meta.json"
# condition_data_root="./custom_novelview"
# video_data_root="./custom_novelview"

# for gpu in 3 4 5 6 7; do
#   CUDA_VISIBLE_DEVICES=$gpu python run_validation.py \
#     --checkpoint_path $checkpoint_path \
#     --pretrained_model_name_or_path $pretrained_model_name_or_path \
#     --meta_file_path $meta_file_path \
#     --condition_data_root $condition_data_root \
#     --video_data_root $video_data_root \
#     --output_dir validation_results/2nodes_28ksteps/novelview \
#     --height 480 \
#     --width 720 \
#     --conditioning_scale 1.0 \
#     --max_num_frames 49 \
#     --mixed_precision bf16 \
#     --target_frames 25 \
#     --total_sections 5 \
#     --section $((gpu-3)) &
# done
