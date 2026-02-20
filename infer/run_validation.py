#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import os
import argparse
import random
import gc
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
from tqdm.auto import tqdm

# PyTorch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as TT
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

# Hugging Face imports
from transformers import AutoTokenizer, T5EncoderModel

# Diffusers imports
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
    CogvideoXBranchModel,
    CogVideoXI2VDualInpaintPipeline
)
from diffusers.utils import export_to_video


def get_args():
    parser = argparse.ArgumentParser(description="Validation script for VideoPainter")
    
    # Model and checkpoint
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory containing the branch model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the lora file"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files"
    )
    
    # Data
    parser.add_argument(
        "--meta_file_path",
        type=str,
        required=True,
        help="Path to validation meta data file"
    )
    parser.add_argument(
        "--condition_data_root",
        type=str,
        required=True,
        help="Path to condition video root directory"
    )
    parser.add_argument(
        "--video_data_root",
        type=str,
        required=True,
        help="Path to ground truth video root directory"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_outputs",
        help="Directory to save validation outputs"
    )
    
    # Video params
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Video width"
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=49,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for saving videos"
    )
    
    # Generation params
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        help="Whether to use dynamic cfg"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Branch model params
    parser.add_argument(
        "--branch_layer_num",
        type=int,
        default=4,
        help="Number of layers in the branch"
    )
    parser.add_argument(
        "--add_first",
        action="store_true",
        help="Enable add_first feature"
    )
    parser.add_argument(
        "--mask_add",
        action="store_true",
        help="Enable mask_add feature"
    )
    parser.add_argument(
        "--mask_background",
        action="store_true",
        help="Enable mask_background feature"
    )
    parser.add_argument(
        "--wo_text",
        action="store_true",
        help="Disable text conditioning"
    )
    parser.add_argument(
        "--first_frame_gt",
        action="store_true",
        help="Use first frame from ground truth"
    )
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="Conditioning scale for classifier-free guidance"
    )
    
    # Performance
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable memory efficient attention with xformers"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int, 
        default=0,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Mixed precision type"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster inference"
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=25,
        help="Number of frames to target and save"
    )
    
    # Add these new arguments
    parser.add_argument(
        "--total_sections",
        type=int,
        default=1,
        help="Total number of sections to split the dataset into"
    )
    parser.add_argument(
        "--section",
        type=int,
        default=0,
        help="Which section to process (0-based index)"
    )
    
    return parser.parse_args()


class VideoInpaintingDataset(Dataset):
    def __init__(
        self,
        meta_file_path,
        condition_data_root,
        video_data_root,
        max_num_frames=49,
        selected_indices=None
    ):
        super().__init__()
        self.meta_file_path = Path(meta_file_path)
        self.condition_data_root = Path(condition_data_root)
        self.video_data_root = Path(video_data_root)
        self.max_num_frames = max_num_frames
        
        if not self.meta_file_path.exists():
            raise ValueError(f"Meta file does not exist: {self.meta_file_path}")
        if not self.video_data_root.exists():
            raise ValueError(f"Instance videos root folder does not exist: {self.video_data_root}")
        if not self.condition_data_root.exists():
            raise ValueError(f"Condition videos root folder does not exist: {self.condition_data_root}")
        
        self._load_dataset(selected_indices)
        
    def _load_dataset(self, selected_indices=None):
        try:
            metadata = json.load(open(self.meta_file_path, "r"))
        except:
            metadata = []
            with open(self.meta_file_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line))
        
        # Use default prompt if needed
        self.prompts = ["Moving view of a city scene." for _ in range(len(metadata))]
        self.videos = [[self.video_data_root / gt_frame_path for gt_frame_path in sample["gt_frames"]] for sample in metadata]
        self.condition_videos = [[self.condition_data_root / render_frame_path for render_frame_path in sample["render_frames"]] for sample in metadata]
        self.scene_idxs = [sample["scene_idx"] for sample in metadata]

        # cast to str
        self.scene_idxs = [f'{scene_idx:06d}' if isinstance(scene_idx, int) else scene_idx for scene_idx in self.scene_idxs]

        
        # Filter by selected indices if provided
        if selected_indices is not None:
            self.prompts = [p for i, p in enumerate(self.prompts) if i in selected_indices]
            self.videos = [v for i, v in enumerate(self.videos) if i in selected_indices]
            self.condition_videos = [c for i, c in enumerate(self.condition_videos) if i in selected_indices]
            self.scene_idxs = [s for i, s in enumerate(self.scene_idxs) if i in selected_indices]
        
        self.bad_index = []

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        if index in self.bad_index:
            print(f"Bad index: {index}")
            return self.__getitem__(random.randint(0, len(self.videos) - 1))
        
        prompt = self.prompts[index]
        video_frames_paths = self.videos[index]
        condition_video_frames_paths = self.condition_videos[index]
        scene_idx = self.scene_idxs[index]

        try:
            video_frames = [cv2.imread(str(frame_path)) for frame_path in video_frames_paths]
            condition_video_frames = [cv2.imread(str(frame_path)) for frame_path in condition_video_frames_paths]
        except Exception as e:
            print(f"Error loading video frames: {e}")
            self.bad_index.append(index)
            return self.__getitem__(random.randint(0, len(self.videos) - 1))

        video = np.array(video_frames)
        condition_video = np.array(condition_video_frames)
        
        return {
            "prompt": prompt,
            "video": video,
            "condition_video": condition_video,
            "scene_idx": scene_idx,
        }


class MyWebDataset:
    def __init__(
        self,
        resolution,
        max_num_frames,
        max_sequence_length=226,
        proportion_empty_prompts=0,
        first_frame_gt=False,
        video_reshape_mode="resize"
    ):
        self.resolution = resolution
        self.max_num_frames = max_num_frames
        self.max_sequence_length = max_sequence_length
        self.proportion_empty_prompts = proportion_empty_prompts
        self.first_frame_gt = first_frame_gt
        self.video_reshape_mode = video_reshape_mode
        self.resolutions = (max_num_frames, resolution[0], resolution[1])
    def _find_nearest_resolution(self, height, width):
        nearest_res = [self.resolutions[0], self.resolutions[1], self.resolutions[2]]
        return nearest_res[1], nearest_res[2]

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def __call__(self, examples):
        pixel_values = []
        conditioning_pixel_values = []
        input_ids = []
        scene_idxs = []
            
        for example in examples:
            caption = example["prompt"]
            video = example["video"]  # frame, height, width, c
            condition_video = example["condition_video"]  # frame, height, width, c
            scene_idx = example["scene_idx"]
            frame, height, width, c = video.shape
            
            # Handle frame count
            if frame > self.max_num_frames:
                begin_idx = 0
                end_idx = begin_idx + self.max_num_frames
                video = video[begin_idx:end_idx]
                condition_video = condition_video[begin_idx:end_idx]
                frame = end_idx - begin_idx
            elif frame <= self.max_num_frames:
                remainder = (3 + (frame % 4)) % 4
                if remainder != 0:
                    video = video[:-remainder]
                    condition_video = condition_video[:-remainder]
                frame = video.shape[0]

                # Pad to max_num_frames frames
                num_repeats = self.max_num_frames - frame
                last_frame = video[-1:]
                repeated_frames = np.repeat(last_frame, num_repeats, axis=0)
                video = np.concatenate([video, repeated_frames], axis=0)

                last_condition_frame = condition_video[-1:]
                repeated_condition_frames = np.repeat(last_condition_frame, num_repeats, axis=0)
                condition_video = np.concatenate([condition_video, repeated_condition_frames], axis=0)

            assert video.shape[0] == self.max_num_frames, f"video shape {video.shape[0]} is not equal to max_num_frames {self.max_num_frames}"
            assert condition_video.shape[0] == self.max_num_frames, f"condition_video shape {condition_video.shape[0]} is not equal to max_num_frames {self.max_num_frames}"

            # Resize video and condition_video
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            condition_video = torch.from_numpy(condition_video).permute(0, 3, 1, 2)
            nearest_res = self._find_nearest_resolution(video.shape[2], video.shape[3])
            
            if self.video_reshape_mode == 'center':
                video_resized = self._resize_for_rectangle_crop(video, nearest_res)
                condition_video_resized = self._resize_for_rectangle_crop(condition_video, nearest_res)
            elif self.video_reshape_mode == 'resize':
                video_resized = torch.stack([resize(frame, nearest_res) for frame in video], dim=0)
                condition_video_resized = torch.stack([resize(frame, nearest_res) for frame in condition_video], dim=0)
            else:
                raise NotImplementedError
                
            video = video_resized
            condition_video = condition_video_resized
            
            if self.first_frame_gt:
                condition_video[0] = video[0]

            # Convert to PIL images for pipeline input
            video_pil = [Image.fromarray(cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)) 
                         for frame in video]
            condition_video_pil = [Image.fromarray(cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)) 
                                   for frame in condition_video]
            
            pixel_values.append(video_pil)
            conditioning_pixel_values.append(condition_video_pil)
            input_ids.append(caption)
            scene_idxs.append(scene_idx)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids,
            "scene_idxs": scene_idxs,
        }


def concatenate_images_horizontally(images1, images2, images3, output_type="np"):
    '''
    Concatenate three lists of images horizontally.
    Args:
        images1: List[Image.Image] or List[np.ndarray]
        images2: List[Image.Image] or List[np.ndarray]
        images3: List[Image.Image] or List[np.ndarray]
    Returns:
        List[Image.Image] or List[np.ndarray]
    '''
    concatenated_images = []
    for img1, img2, img3 in zip(images1, images2, images3):
        # Convert images to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        arr3 = np.array(img3)

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate((arr1, arr2, arr3), axis=1)

        # Convert back to PIL Image if needed
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        concatenated_images.append(concatenated_img)
    return concatenated_images


def main():
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and weight type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Load models
    print("Loading models...")
    
    # Branch
    # First load from pretrained model
    if os.path.exists(os.path.join(args.checkpoint_path, "branch")):
        branch = CogvideoXBranchModel.from_pretrained(os.path.join(args.checkpoint_path, "branch"), torch_dtype=weight_dtype).to(dtype=weight_dtype).cuda()
        print(f"Loading branch model from {args.checkpoint_path}")
        if args.lora_path is None:
            pipe = CogVideoXI2VDualInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                branch=branch,
                torch_dtype=weight_dtype,
            )
        else:
            print(f"Loading the lora from: {args.lora_path}")
            # load the transformer
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=weight_dtype,
                id_pool_resample_learnable=False,
            ).to(dtype=weight_dtype).cuda()

            pipe = CogVideoXI2VDualInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                branch=branch,
                transformer=transformer,
                torch_dtype=weight_dtype,
            )
        
            pipe.load_lora_weights(
                args.lora_path, 
                weight_name="pytorch_lora_weights.safetensors", 
                adapter_name="test_1",
                target_modules=["transformer"]
                )
            # pipe.fuse_lora(lora_scale=1 / lora_rank)

            list_adapters_component_wise = pipe.get_list_adapters()
            print(f"list_adapters_component_wise: {list_adapters_component_wise}")
    else:
        raise ValueError(f"Branch model not found in {args.checkpoint_path}")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        ).to(device=device, dtype=weight_dtype).cuda()
        branch = CogvideoXBranchModel.from_transformer(
            transformer=transformer,
            num_layers=1,
            attention_head_dim=transformer.config.attention_head_dim,
            num_attention_heads=transformer.config.num_attention_heads,
            load_weights_from_transformer=True
        ).to(dtype=weight_dtype).cuda()
        pipe = CogVideoXI2VDualInpaintPipeline.from_pretrained(
            model_path,
            branch=branch,
            transformer=transformer,
            torch_dtype=weight_dtype,
            # device_map='balanced',
        )
        print("Intializing model from pretrained model")

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")

    # if long_video:
    
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            print("xformers is not available. Make sure it is installed correctly")

    # load the json to check the length
    with open(args.meta_file_path, 'r') as f:
        metadata = json.load(f)
    print(f"Total number of samples: {len(metadata)}")

    # scan through the metadata to check if it exists in the output directory
    valid_indices = []
    for i, sample in enumerate(metadata):
        # check!
        if os.path.exists(os.path.join(args.output_dir, 'frames', f"sample_{sample['scene_idx']}")) and len(os.listdir(os.path.join(args.output_dir, 'frames', f"sample_{sample['scene_idx']}"))) > 24:
            print(f"Skipping sample {args.output_dir}/frames/sample_{sample['scene_idx']} because it already exists")
            continue
        valid_indices.append(i)
    
    # Calculate indices for the selected section
    num_samples = len(valid_indices)
    part_size = (num_samples + args.total_sections - 1) // args.total_sections  # Ceiling division
    start_idx = args.section * part_size
    end_idx = min((args.section + 1) * part_size, num_samples)  # Ensure we don't go beyond the last sample
    
    selected_indices = valid_indices[start_idx:end_idx]
    print(f"Processing section {args.section+1}/{args.total_sections}: Samples {start_idx} to {end_idx-1} ({end_idx - start_idx} samples)")
    
    # Create dataset for this section
    validation_dataset = VideoInpaintingDataset(
        meta_file_path=args.meta_file_path,
        condition_data_root=args.condition_data_root,
        video_data_root=args.video_data_root,
        max_num_frames=args.max_num_frames,
        selected_indices=selected_indices
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MyWebDataset(
            resolution=[args.height, args.width],
            max_num_frames=args.max_num_frames,
            first_frame_gt=args.first_frame_gt,
            video_reshape_mode="center",
        ),
        pin_memory=True,
        num_workers=args.num_workers,
    )
    
    # Set up generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    
    # Run inference
    print(f"Running inference on {len(validation_dataset)} validation samples...")
    for i, batch in enumerate(tqdm(validation_dataloader)):

        # check!
        if os.path.exists(os.path.join(args.output_dir, 'frames', f"sample_{batch['scene_idxs'][0]}")) and len(os.listdir(os.path.join(args.output_dir, 'frames', f"sample_{batch['scene_idxs'][0]}"))) > 24:
            print(f"Skipping sample {args.output_dir}/frames/sample_{batch['scene_idxs'][0]} because it already exists")
            continue

        # Set up pipeline args
        pipeline_args = {
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "image": batch["pixel_values"][0][0],
            "video": batch["pixel_values"],
            "prompt": batch["input_ids"],
            "condition_video": batch["conditioning_pixel_values"],
            "num_frames": len(batch["pixel_values"][0]),
            "strength": 1.0,
            "conditioning_scale": args.conditioning_scale,
            "mask_background": args.mask_background,
            "add_first": args.add_first,
            "wo_text": args.wo_text,
            "mask_add": args.mask_add,
            "replace_gt": True,  # Try both options
        }
        
        # Run inference
        output = pipe(
            **pipeline_args,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            output_type="np"
        )
        
        # Get generated video frames
        generated_frames = output.frames[0]
        
        # Process original video and condition video
        original_video = pipe.video_processor.preprocess_video(
            pipeline_args['video'][0], 
            height=generated_frames.shape[1], 
            width=generated_frames.shape[2]
        )
        
        condition_video = pipe.video_processor.preprocess_video(
            pipeline_args['condition_video'][0], 
            height=generated_frames.shape[1], 
            width=generated_frames.shape[2]
        )
        
        original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
        condition_video = pipe.video_processor.postprocess_video(video=condition_video, output_type="np")[0]

        # select target frames
        if args.target_frames is not None:  
            original_video = original_video[:args.target_frames]
            condition_video = condition_video[:args.target_frames]
            generated_frames = generated_frames[:args.target_frames]
        
        # Concatenate horizontally
        concatenated_frames = concatenate_images_horizontally(
            original_video, 
            condition_video, 
            generated_frames
        )
        
        # Save as video
        scene_idx = batch["scene_idxs"][0]
        video_filename = os.path.join(args.output_dir, 'cat_videos', f"sample_{scene_idx}.mp4")
        os.makedirs(os.path.join(args.output_dir, 'cat_videos'), exist_ok=True)
        export_to_video(concatenated_frames, video_filename, fps=args.fps)
        
        # Save individual frames
        frames_dir = os.path.join(args.output_dir, 'frames', f"sample_{scene_idx}")
        os.makedirs(frames_dir, exist_ok=True)
            
        for j, frame in enumerate(generated_frames):
            frame_filename = os.path.join(frames_dir, f"frame_{j}.png")
            # Convert from float32 to uint8
            frame = (frame * 255).astype(np.uint8)
            Image.fromarray(frame).save(frame_filename)
        
        # Clean up to avoid memory issues
        del generated_frames, original_video, condition_video, concatenated_frames
        torch.cuda.empty_cache()
        gc.collect()
            
        # except Exception as e:
        #     print(f"Error processing sample {i}: {e}")
        #     continue
    
    print(f"Validation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()