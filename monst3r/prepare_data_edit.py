import sys
import time
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path
from torchvision.utils import save_image
import json
import random

from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime
from tqdm import tqdm

global json_path, data_root

json_path = None
data_root = None

def process_instance_sequence(frames, corners_dict, img_size=(1600, 900), drop_crop=False, device='cuda'):
    """
    Process each instance: crop from first frame and replant in subsequent frames
    
    Args:
        frames: tensor of shape (25, H, W, 3) or (25, 576, 1024, 3)
        corners_dict: dict of {instance_id: corners array (25, 8, 2)} in 1600x900 scale
        img_size: tuple of original corner coordinates scale
    Returns:
        processed_frames: frames with replanted instances
    """
    # Convert frames to GPU tensor if needed
    if not torch.is_tensor(frames):
        frames = torch.from_numpy(frames).to(device)
    elif frames.device.type != 'cuda':
        frames = frames.to(device)
        
    processed_frames = frames.clone()
    num_frames, H, W = frames.shape[:3]
    
    # Calculate scaling factors
    scale_x = W / img_size[0]
    scale_y = H / img_size[1]
    
    for instance_id, corners in corners_dict.items():
        # Get first frame corners and move to GPU
        first_frame_corners = corners[0][0].clone().T  # (8, 2)
        first_frame_corners[:, 0] *= scale_x
        first_frame_corners[:, 1] *= scale_y
        
        # Convert 8 corners to 4-point bbox
        first_bbox = corners_to_bbox(first_frame_corners, device=device)
        
        # Crop the instance from first frame
        instance_crop = crop_bbox(frames[0], first_bbox, device=device)

        if drop_crop:
            # Generate drop percentages on GPU
            drop_percentages = torch.linspace(0, 0.5, num_frames-1, device=device)
        else:
            drop_percentages = torch.zeros(num_frames-1, device=device)
        
        # Replant in each frame
        for frame_idx in range(1, num_frames):
            if corners[frame_idx] is None:
                continue
            current_corners = corners[frame_idx][0].clone().T
            current_corners[:, 0] *= scale_x
            current_corners[:, 1] *= scale_y
            current_bbox = corners_to_bbox(current_corners, device=device)
            
            # Paste the cropped instance
            processed_frames[frame_idx] = paste_cropped_area(
                processed_frames[frame_idx], 
                instance_crop, 
                current_bbox,
                drop_percentage=drop_percentages[frame_idx-1].item(),
                device=device
            )
    
    return processed_frames

def corners_to_bbox(corners, device='cuda'):
    """Convert 8x2 corners to 4x2 bbox"""
    # Convert numpy array to tensor if needed and move to GPU
    if not torch.is_tensor(corners):
        corners = torch.from_numpy(corners).to(device)
    elif corners.device.type != 'cuda':
        corners = corners.to(device)
    
    x_min = torch.min(corners[:, 0])
    x_max = torch.max(corners[:, 0])
    y_min = torch.min(corners[:, 1])
    y_max = torch.max(corners[:, 1])
    
    return torch.tensor([
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max]   # bottom-left
    ], device=device)

def crop_bbox(image, bbox, device='cuda'):
    """Crop image region defined by bbox"""
    if not torch.is_tensor(image):
        image = torch.from_numpy(image).to(device)
    elif image.device.type != 'cuda':
        image = image.to(device)
        
    if not torch.is_tensor(bbox):
        bbox = torch.from_numpy(bbox).to(device)
    elif bbox.device.type != 'cuda':
        bbox = bbox.to(device)
        
    x_min = int(torch.min(bbox[:, 0]).item())
    x_max = int(torch.max(bbox[:, 0]).item())
    y_min = int(torch.min(bbox[:, 1]).item())
    y_max = int(torch.max(bbox[:, 1]).item())
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    
    return image[y_min:y_max, x_min:x_max].clone()

def paste_cropped_area(target_image, cropped_area, target_bbox, drop_percentage=0., device='cuda'):
    """Resize and paste cropped area with smooth blending"""
    # Move all inputs to GPU
    if not torch.is_tensor(target_image):
        target_image = torch.from_numpy(target_image).to(device)
    elif target_image.device.type != 'cuda':
        target_image = target_image.to(device)
        
    if not torch.is_tensor(cropped_area):
        cropped_area = torch.from_numpy(cropped_area).to(device)
    elif cropped_area.device.type != 'cuda':
        cropped_area = cropped_area.to(device)
        
    if not torch.is_tensor(target_bbox):
        target_bbox = torch.from_numpy(target_bbox).to(device)
    elif target_bbox.device.type != 'cuda':
        target_bbox = target_bbox.to(device)
    
    # Get target dimensions
    original_x_min = target_x_min = int(torch.min(target_bbox[:, 0]).item())
    original_x_max = target_x_max = int(torch.max(target_bbox[:, 0]).item())
    original_y_min = target_y_min = int(torch.min(target_bbox[:, 1]).item())
    original_y_max = target_y_max = int(torch.max(target_bbox[:, 1]).item())
    original_width = original_x_max - original_x_min
    original_height = original_y_max - original_y_min
    
    # Ensure coordinates are within bounds
    target_x_min = max(0, target_x_min)
    target_y_min = max(0, target_y_min)
    target_x_max = min(target_image.shape[1], target_x_max)
    target_y_max = min(target_image.shape[0], target_y_max)
    
    # Calculate target size
    target_width = target_x_max - target_x_min
    target_height = target_y_max - target_y_min
    
    if target_width <= 0 or target_height <= 0:
        return target_image
    
    # Resize cropped area using torch interpolate
    if (target_width, target_height) != (original_width, original_height):
        resized_crop = F.interpolate(
            cropped_area.permute(2, 0, 1).unsqueeze(0),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # Determine valid region
        crop_x_start = max(0, target_x_min - original_x_min)
        crop_x_end = crop_x_start + target_width
        crop_y_start = max(0, target_y_min - original_y_min)
        crop_y_end = crop_y_start + target_height
        
        resized_crop = resized_crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    else:
        resized_crop = F.interpolate(
            cropped_area.permute(2, 0, 1).unsqueeze(0),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

    # Random pixel dropping using torch
    if drop_percentage > 0:
        mask_drop = torch.rand(resized_crop.shape[:2], device=device) > drop_percentage
        if resized_crop.ndim == 3:
            mask_drop = mask_drop.unsqueeze(-1).expand(-1, -1, resized_crop.shape[2])
        resized_crop = resized_crop * mask_drop.float()
    
    # Create result image and mask
    result_image = target_image.clone()
    mask = torch.ones(resized_crop.shape[:2], device=device)
    
    # Apply Gaussian blur to mask
    mask = mask.unsqueeze(0).unsqueeze(0)
    gaussian_kernel = torch.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], device=device).float() / 256.0
    gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5)
    mask = F.conv2d(mask, gaussian_kernel, padding=2)
    mask = mask.squeeze()
    
    try:
        # Blend images
        roi = result_image[target_y_min:target_y_max, target_x_min:target_x_max]
        mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        blended = (resized_crop * mask + roi * (1 - mask))
        
        result_image[target_y_min:target_y_max, target_x_min:target_x_max] = blended
    except Exception as e:
        print(f"Error in blending paste_cropped_area: {e}")
        print(f"target_y_min: {target_y_min}, target_y_max: {target_y_max}, target_x_min: {target_x_min}, target_x_max: {target_x_max}")
        
    return result_image

def resize_and_reshape_image(image_tensor):
    """
    Resize and reshape image tensor from (900, 1600, 3) to (1, 576, 1024, 3)
    
    Args:
        image_tensor: tensor of shape (H, W, 3)
    Returns:
        resized_tensor: tensor of shape (1, 576, 1024, 3)
    """
    # 1. Add batch and permute to (1, 3, H, W) for F.interpolate
    x = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 900, 1600)
    
    # 2. Resize the image
    resized = F.interpolate(
        x, 
        size=(576, 1024), 
        mode='bilinear',
        align_corners=False
    )
    
    # 3. Permute back to (1, H, W, 3)
    reshaped = resized.permute(0, 2, 3, 1)

    return reshaped

class ViewCrafter_DataEngine:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        
    def run_dust3r(self, input_images,clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)
        
        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

        pc = self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)[0].detach()

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):
      
        # imgs = to_numpy(imgs)
        # pts3d = to_numpy(pts3d)

        if masks is None:
            # pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            # col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
            if isinstance(pts3d, torch.Tensor):
                pts = pts3d.to(device)
            else:
                pts = torch.from_numpy(pts3d).to(device)
            if isinstance(imgs, torch.Tensor):
                col = imgs.to(device)
            else:
                col = torch.from_numpy(imgs).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views).to(self.device)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None

        return images, view_masks
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer'].to(self.device)
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views,renderer,self.device,nbv=False)
        return render_results, viewmask

    def generate_traj_specified_single(self, c2ws_anchor,H,W,fs,c,theta=0., phi=0.,d_r=0.,d_x=0.,d_y=0.,frame=1,device='cuda'):
        # Initialize a camera.
        """
        The camera coordinate sysmte in COLMAP is right-down-forward
        Pytorch3D is left-up-forward
        """

        c2w_new = sphere2pose(c2ws_anchor, np.float32(theta), np.float32(phi), np.float32(d_r), device, np.float32(d_x),np.float32(d_y))
        c2ws = c2w_new
        num_views = c2ws.shape[0]

        R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
        ## 将dust3r坐标系转成pytorch3d坐标系
        R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
        new_c2w = torch.cat([R, T], 2)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
        R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
        image_size = ((H, W),)  # (h, w)
        cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
        return cameras,num_views
    
    def nvs_multi_pose_fixed_view_change_objects(self, images, img_ori, target_frames, device, save_dir=None, instance_corners=None, phi=0., theta=0., d_r=0., d_x=0., d_y=0.):
        """
        Render multiple camera poses from a fixed viewing transformation
        
        Args:
            scene: Scene object containing camera parameters and point clouds
            images: Dictionary of input images
            img_ori: Original input images
            target_frames: List of target frame indices
            sampled_indices: List of sampled frame indices
            opts: Options/configuration object
            device: Torch device to use
            c2ws: Camera poses in world coordinates [N, 4, 4]
            intrinsics: Camera intrinsics matrix [3, 3]
            save_dir: Directory to save outputs
            phi: Fixed view transformation parameter for rotation around y-axis
            theta: Fixed view transformation parameter for rotation around x-axis 
            d_r: Fixed view transformation parameter for radius change
            d_x: Fixed view transformation parameter for x translation
            d_y: Fixed view transformation parameter for y translation
            gradio: Boolean flag for gradio interface
        """
        # Get all camera poses and parameters
        c2ws = self.scene.get_im_poses().detach()
        # Use only first frame's camera intrinsics
        principal_points = self.scene.get_principal_points()[0:1].detach()
        focals = self.scene.get_focals()[0:1].detach()

        # Get image dimensions
        shape = images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])

        print(f'true shape H: {H}, W: {W}')
        
        # Get point clouds and depth
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]

        
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[0][H//2,W//2]  # Use first frame's depth
        radius = depth_avg*self.opts.center_scale

        # Transform coordinates to object-centric
        c2ws, pcd = world_point_to_obj(
            poses=c2ws, 
            points=torch.stack(pcd), 
            k=0,  # Use first frame as reference
            r=radius, 
            elevation=self.opts.elevation, 
            device=device
        )

        imgs = np.array(self.scene.imgs)

        print(f'imgs shape: {imgs.shape}')
        masks = None

        # Apply the fixed view transformation to all camera poses
        camera_poses = []
        for i in range(len(c2ws)):
            # Generate single transformed pose
            single_pose, _ = self.generate_traj_specified_single(
                c2ws[i:i+1],  # Take one pose at a time
                H, W,
                focals,  # Use first frame's focal length for all
                principal_points,  # Use first frame's principal point for all
                theta, phi, d_r, d_x, d_y,
                1,  # Only generate one frame per pose
                device
            )
            camera_poses.append(single_pose)

        # Combine all transformed cameras
        camera_traj = camera_poses[0]
        for cam in camera_poses[1:]:
            camera_traj.R = torch.cat([camera_traj.R, cam.R])
            camera_traj.T = torch.cat([camera_traj.T, cam.T])

        # Concatenate all point clouds and images
        # all_pcd = torch.cat([p.reshape(-1, 3) for p in pcd], dim=0)
        # all_imgs = np.concatenate([img.reshape(-1, 3) for img in imgs], axis=0)

        def black_out_instance_cv(point_cloud, corners_dict, img_size=(1600, 900)):
            """
            Black out instances in point cloud using scaled corners
            
            Args:
                point_cloud: tensor of shape (H, W, 3)
                corners_dict: dict of {instance_id: corners array (25, 8, 2)} in 1600x900 scale
                img_size: tuple of (width, height) of original corner coordinates
            Returns:
                modified_pc: point cloud with blacked out instances
            """
            # Ensure point cloud is on the same device
            device = point_cloud.device
            modified_pc = point_cloud.clone()  # Create a copy of the point cloud
            H, W = modified_pc.shape[:2]
            
            # Calculate scaling factors
            scale_x = W / img_size[0]
            scale_y = H / img_size[1]
            
            for instance_id, corners in corners_dict.items():
                # Get first frame corners
                first_frame_corners = corners[0][0].clone().T  # (8, 2)
                
                first_frame_corners[:, 0] *= scale_x
                first_frame_corners[:, 1] *= scale_y
                
                # Get bbox coordinates and scale them using torch
                x_min = int(torch.min(first_frame_corners[:, 0]).item())
                x_max = int(torch.max(first_frame_corners[:, 0]).item())
                y_min = int(torch.min(first_frame_corners[:, 1]).item())
                y_max = int(torch.max(first_frame_corners[:, 1]).item())
                
                # Ensure coordinates are within bounds
                x_min = max(0, x_min)
                x_max = min(W, x_max)
                y_min = max(0, y_min)
                y_max = min(H, y_max)
                
                # Black out the bbox region
                modified_pc[y_min:y_max, x_min:x_max] = 0
            
            return modified_pc

        os.makedirs(save_dir, exist_ok=True)
        #### 
        if len(instance_corners) > 0:
            pcd[0] = black_out_instance_cv(pcd[0], instance_corners)
        # Render from all transformed viewpoints
        render_results, viewmask = self.run_render(
            pcd[0].reshape(-1, 3),#all_pcd,
            imgs[0].reshape(-1, 3),#all_imgs, 
            masks,
            H, W,
            camera_traj,
            len(c2ws),
            device
        )

        # Resize renders to target resolution
        render_results = F.interpolate(
            render_results.permute(0,3,1,2), 
            size=(576, 1024), 
            mode='bilinear', 
            align_corners=False
        ).permute(0,2,3,1)
            
        # Save outputs
        # save_video(render_results, os.path.join(save_dir, 'original_render.mp4'))
        # save_frames(render_results, save_dir, save_names=[f'original_base{target_frames[0]}_idx{idx}' for idx in sampled_indices], folder=None)
        # save_pointcloud_with_normals(
        #     imgs[0].reshape(-1, 3),#all_imgs,
        #     pcd[0].reshape(-1, 3),#all_pcd,
        #     msk=None,
        #     save_path=os.path.join(save_dir,'pcd_all.ply'),
        #     mask_pc=False,
        #     reduce_pc=False
        # )
        # print(f'saved to {save_dir}')
        
        
        return render_results
        
    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)
        self.scene = None

    def process_single_scene(self, scene_data, start_idx=None, interval=None):
        """
        Process a single scene - extracted from process_dataset for parallel execution
        """
        start_time = time.time()
        idx, scene, data_root = scene_data
        
        target_frames = [0]
        self.target_frames = target_frames

        frame_files = scene['frames']
        frame_paths = [os.path.join(data_root, p) for p in frame_files]
        instance_corners = scene['instance_corners']
            
        save_dir = os.path.join(self.opts.save_dir, f'sample_{idx:06d}')
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 25:
            print(f'skip {save_dir}')
            return None, None
        
        os.makedirs(save_dir, exist_ok=True)
    
        images = load_images(frame_paths, size=512, force_1024=True)
        img_ori = [(images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. for i in range(len(images))]

        self.images = images
        self.img_ori = img_ori
        self.run_dust3r(input_images=images, clean_pc=True)
        # self.nvs_multi_pose_fixed_view(c2ws=None, intrinsics=camera_intrinsics, save_dir=save_dir) # this is for dust3r pose
        rendered_results = self.nvs_multi_pose_fixed_view_change_objects(images, img_ori, target_frames, self.device, save_dir, instance_corners=instance_corners) # this is for nuscenes pose

        self.scene = None
        torch.cuda.empty_cache()

        firstframe_img_ori = resize_and_reshape_image(img_ori[0])
        rendered_results[0] = firstframe_img_ori
        if len(instance_corners) > 0:
            print('encountered instance corners')
            # return None, None
            processed_frames = process_instance_sequence(rendered_results, instance_corners)
        else:
            processed_frames = rendered_results
        
        sampled_indices = range(25)
        # save_video(processed_frames, os.path.join(save_dir, 'edited_render.mp4'))
        save_frames(processed_frames, save_dir, save_names=[f'edited_base{target_frames[0]}_idx{idx}' for idx in sampled_indices], folder=None)

        print(f'data saved in {save_dir}')
        # save_video(torch.stack(img_ori), os.path.join(save_dir, 'frames.mp4'))

        # scene.update({'render_frames': [f'sample_{idx}/edited_base{target_frames[0]}_idx{idx}' for idx in sampled_indices]})

        # # Convert NumPy arrays to lists and handle None sublists
        # for key in scene['instance_corners']:
        #     scene['instance_corners'][key] = [
        #         [array.tolist() for array in sublist] if sublist is not None else None
        #         for sublist in scene['instance_corners'][key]
        #     ]
        
        used_time = time.time() - start_time
        

        return scene, used_time


    def process_dataset(self, json_path, data_root, output_dir):
        """
        Process entire dataset sequentially and save results
        """
       
        frame_cam_corners_samples = torch.load('../new_frame_cam_corners_samples.pth')
        frame_cam_samples_indices = torch.load('../frame_cam_samples_indices.pth')

        rest_frame_cam_corners_samples = []
        rest_frame_cam_samples_indices = []
        for i in range(len(frame_cam_corners_samples)):
            idx = frame_cam_samples_indices[i]
            save_dir = os.path.join(self.opts.save_dir, f'sample_{idx:06d}')
            if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 25:
                print(f'skip {save_dir}')
                continue
            rest_frame_cam_corners_samples.append(frame_cam_corners_samples[i])
            rest_frame_cam_samples_indices.append(idx)

        # Define the function to split and extract nth split
        def split_and_get_nth(data, n, splits=16):
            if n < 0 or n >= splits:
                raise ValueError(f"Invalid split index {n}. Must be between 0 and {splits - 1}.")
            
            # Calculate the size of each split
            split_size = len(data) // splits
            remainder = len(data) % splits  # For uneven splitting
            
            # Create the splits
            splits_indices = []
            start_idx = 0
            for i in range(splits):
                extra = 1 if i < remainder else 0  # Add extra for first 'remainder' splits
                splits_indices.append((start_idx, start_idx + split_size + extra))
                start_idx += split_size + extra
            
            start, end = splits_indices[n]
            return data[start:end]

        split_corners = split_and_get_nth(rest_frame_cam_corners_samples, self.opts.section, self.opts.total_sections)
        split_indices = split_and_get_nth(rest_frame_cam_samples_indices, self.opts.section, self.opts.total_sections)

        # Prepare data for sequential processing
        scene_data = [(idx, scene, data_root) for idx, scene in zip(split_indices, split_corners)]

        training_samples = []

        times = list()
        for data in tqdm(scene_data, desc='Processing scenes'):  # Limited to 4 scenes as in original

            try:
                result, used_time = self.process_single_scene(data)
                if result is None:
                    continue
                training_samples.append(result)
                times.append(used_time)
                avg_time = sum(times) / len(times)
                total_time = avg_time * (len(scene_data) - len(times))
                print(f'Split:{self.opts.section}, time used: {times[-1]:.2f}s, estimated left time: {total_time/3600:.2f}h')
            except Exception as e:
                print(f'Error: {e}')
                continue

        # Save metadata
        # with open(os.path.join(os.path.dirname(json_path), 'nuScenes_frontview_base0_fullframes_rendered.json'), 'a') as f:
        #     json.dump(training_samples, f)

# Example usage:
# render_results = render_dust3r_scene(scene)
# save_video(render_results, 'rendered_views.mp4')

if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir,exist_ok=True)
    pvd = ViewCrafter_DataEngine(opts)

    json_path = '/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_frontview_caminex_fullframes.json'
    data_root = "/lpai/dataset/nuscenes-imgs/0-1-0/nuscenes"

    pvd.process_dataset(json_path, data_root, opts.save_dir)