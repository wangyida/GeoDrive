# CUDA_VISIBLE_DEVICES=2 python render.py \
#     --output_dir outputs/nuscenes_train_plain_stride1window3 \
#     --device 'cuda' \
#     --batch_size 32 \
#     --image_size 512 \
#     --total_sections 1 \
#     --section 0

for gpu in 4 5 6 7; do
    section=$((gpu * 1))
    CUDA_VISIBLE_DEVICES=$gpu python render.py \
        --output_dir outputs/nuscenes_train_plain_stride1window3 \
        --device 'cuda' \
        --batch_size 32 \
        --image_size 512 \
        --total_sections 16 \
        --section $section > ../logs/log_${gpu}_${section}.out 2>&1 &
    pids+=($!)  # Record the process ID
done

# for gpu in 0 1 2 3 4 5 6 7; do
#     for process in {0..1}; do
#         section=$((gpu * 2 + process))
#         CUDA_VISIBLE_DEVICES=$gpu python render.py \
#             --output_dir outputs/nuscenes_train_plain \
#             --device 'cuda' \
#             --batch_size 16 \
#             --image_size 512 \
#             --total_sections 16 \
#             --section $section > ../logs/log_${gpu}_${section}.out 2>&1 &
#         pids+=($!)  # Record the process ID
#     done
# done