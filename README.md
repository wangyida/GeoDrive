# GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control

**GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control**<br>

Abstract: Recent advancements in world models have revolutionized dynamic environment simulation, allowing systems to foresee future states and assess potential actions. In autonomous driving, these capabilities help vehicles anticipate the behavior of other road users, perform risk-aware planning, accelerate training in simulation, and adapt to novel scenarios, thereby enhancing safety and reliability. Current approaches exhibit deficiencies in maintaining robust 3D geometric consistency or accumulating artifacts during occlusion handling, both critical for reliable safety assessment in autonomous navigation tasks. To address this, we introduce GeoDrive, which explicitly integrates robust 3D geometry conditions into driving world models to enhance spatial understanding and action controllability. Specifically, we first extract a 3D representation from the input frame and then obtain its 2D rendering based on the user-specified ego-car trajectory. To enable dynamic modeling, we propose a dynamic editing module during training to enhance the renderings by editing the positions of the vehicles. Extensive experiments demonstrate that our method significantly outperforms existing models in both action accuracy and 3D spatial awareness, leading to more realistic, adaptable, and reliable scene modeling for safer autonomous driving. Additionally, our model can generalize to novel trajectories and offers interactive scene editing capabilities, such as object editing and object trajectory control.

**Anthony Chen<sup>1,2\*</sup>**, **Wenzhao Zheng*<sup>3\*</sup>**, **Yida Wang<sup>2\*</sup>**, **Xueyang Zhang<sup>2</sup>**, **Kun Zhan<sup>2</sup>**, **Peng Jia<sup>2</sup>**, **Kurt Keutzer<sup>3</sup>**, **Shanghang Zhang<sup>1†</sup>**  
<sup>1</sup>Peking University  <sup>2</sup>Li Auto Inc.  <sup>3</sup>UC Berkeley  
<br>
\* indicates equal contribution  
† corresponding author  


**[Paper](https://arxiv.org/abs/2505.22421)**

[![Watch the video](https://img.youtube.com/vi/LECkvCff6v0/0.jpg)](https://www.youtube.com/watch?v=LECkvCff6v0)

## TODO List

- **[2025-05-30]** ✅ Release [paper](https://arxiv.org/abs/2505.22421)
- **[2026-02-20]** ✅ Release inference code
- **[2026-02-20]** ✅ Release model checkpoints

## Getting Started

<summary><b>Environment Requirement</b></summary>

We recommend first use `conda` to create virtual environment, and install needed libraries.

```bash
conda create -n geodrive python=3.10 -y
conda activate geodrive
pip install -r requirements.txt
```

Then, install custom diffusers with:

```bash
cd ./diffusers
pip install -e .
```

Next, install required ffmpeg:

```bash
conda install -c conda-forge ffmpeg -y
```

<summary><b>Data Preparation</b></summary>


We use processed data from NuScenes. Please follow instruction from another folder 'monst3r' to prepare the data. 


## 🏃🏼 Running Scripts

<summary><b>Training</b></summary>

Train the GeoDrive using the script:

```bash
export MODEL_PATH="THUDM/CogVideoX-5b-I2V"
export CACHE_PATH="~/.cache"
export CONDITION_DATASET_PATH="PATH/TO/CONDITION_TRAIN"
export VIDEO_DATASET_PATH="PATH/TO/NUSCENES"
export VAL_CONDITION_DATASET_PATH=""PATH/TO/CONDITION_VAL"
export PROJECT_NAME="GeoDrive"
export RUNS_NAME="GeoDrive"
export OUTPUT_PATH="./${PROJECT_NAME}/${RUNS_NAME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_LAUNCH_WAIT_TIMEOUT=300
export DS_SKIP_CUDA_CHECK=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=10086 --master_port=$MASTER_PORT \
  train.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --meta_file_path PATH/TO/META_TRAIN \
  --val_meta_file_path PATH/TO/META_VAL \
  --video_data_root $VIDEO_DATASET_PATH \
  --condition_data_root $CONDITION_DATASET_PATH \
  --val_condition_data_root $VAL_CONDITION_DATASET_PATH \
  --dataloader_num_workers 1 \
  --num_validation_videos 1 \
  --validation_epochs 2 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 10 \
  --video_reshape_mode "center" \
  --branch_layer_num 2 \
  --train_batch_size 1 \
  --num_train_epochs 20 \
  --checkpointing_steps 2000 \
  --validating_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 1000 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0.05 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --tracker_name $PROJECT_NAME \
  --runs_name $RUNS_NAME \
  --mix_train_ratio 0 \
  --first_frame_gt 

```

<summary><b>Inference</b></summary>

You can run inference with the script:

```bash
cd infer

pretrained_model_name_or_path="THUDM/CogVideoX-5b-I2V"
checkpoint_path="PATH/TO/CKPT"
meta_file_path="PATH/TO/META"
condition_data_root="PATH/TO/CONDITION"
video_data_root="PATH/TO/NUSCENES"

python run_validation.py \
--checkpoint_path $checkpoint_path \
--pretrained_model_name_or_path $pretrained_model_name_or_path \
--meta_file_path $meta_file_path \
--condition_data_root $condition_data_root \
--video_data_root $video_data_root \
--output_dir /PATH/TO/OUTPUT \
--height 480 \
--width 720 \
--max_num_frames 49 \
--mixed_precision bf16 \
--target_frames 25 \

```


## Citation
```
@misc{chen2025geodrive3dgeometryinformeddriving,
      title={GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control}, 
      author={Anthony Chen and Wenzhao Zheng and Yida Wang and Xueyang Zhang and Kun Zhan and Peng Jia and Kurt Keutzer and Shanghang Zhang},
      year={2025},
      eprint={2505.22421},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.22421}, 
}
```
