from asyncio import tasks
import sys
import time
import os
from altair import Theta
from matplotlib.pyplot import sca
import torch
import numpy as np
import cv2
from PIL import Image
import json
from tqdm import tqdm
from datetime import datetime
import argparse
import functools
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras
import torch.nn.functional as F

# MonST3R imports
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb
import copy

from pvd_utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--model_name', type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--section', type=int, default=0)
    parser.add_argument('--total_sections', type=int, default=16)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--center_scale', type=float, default=1.0)
    parser.add_argument('--dpt_trd', type=float, default=1.)
    parser.add_argument('--bg_trd', type=float, default=0.)
    parser.add_argument('--theta', type=float, default=0.)
    parser.add_argument('--phi', type=float, default=0.)
    parser.add_argument('--d_r', type=float, default=0.)
    parser.add_argument('--d_x', type=float, default=0.)
    parser.add_argument('--d_y', type=float, default=0.)
    parser.add_argument('--elevation', type=float, default=5.)
    parser.add_argument('--min_conf_thr', type=float, default=1.1)
    parser.add_argument('--scenegraph_type', type=str, default='swinstride')
    parser.add_argument('--swinsize', type=int, default=5)
    parser.add_argument('--refid', type=int, default=0)
    return parser

# Add helper function for camera setup
def setup_renderer(cameras, image_size):
    """Setup PyTorch3D renderer"""
    from pytorch3d.renderer import (
        PointsRasterizer,
        PointsRenderer,
        AlphaCompositor,
        PointsRasterizationSettings,
    )
    
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003,
        points_per_pixel=10
    )
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    
    return {"renderer": renderer, "cameras": cameras}

def sphere2pose(c2ws_input, theta, phi, r, device,x=None,y=None):
    # c2ws = copy.deepcopy(c2ws_input) # this produces a bug
    c2ws = c2ws_input.detach().clone().to(device)
    #先沿着世界坐标系z轴方向平移再旋转
    c2ws[:,2,3] += r
    if x is not None:
        c2ws[:,1,3] += y
    if y is not None:
        c2ws[:,0,3] += x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([[1, 0, 0, 0],
                    [0, cos_value_x, -sin_value_x, 0],
                    [0, sin_value_x, cos_value_x, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1).to(device)
    
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([[cos_value_y, 0, sin_value_y, 0],
                    [0, 1, 0, 0],
                    [-sin_value_y, 0, cos_value_y, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1).to(device)
    
    c2ws = torch.matmul(rot_mat_x,c2ws)
    c2ws = torch.matmul(rot_mat_y,c2ws)

    return c2ws 

class MonST3R_DataEngine:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.setup_monst3r()
        
    def setup_monst3r(self):
        """Initialize MonST3R model"""
        if os.path.exists(self.opts.model_path):
            weights_path = self.opts.model_path
        else:
            weights_path = self.opts.model_name
            
        self.model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(self.device)
        self.model.eval()

    def render_pcd(self, pts3d, imgs, masks, views, renderer, device, nbv=False):
        """Render point cloud from different viewpoints"""
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks is None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views).to(device)
        images = renderer(point_cloud)
        
        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts], features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else:
            view_masks = None
            
        return images, view_masks

    def run_render(self, pcd, imgs, masks, H, W, camera_traj, num_views, nbv=False):
        """Run rendering pipeline"""
        render_setup = setup_renderer(camera_traj, image_size=(H, W))
        renderer = render_setup['renderer'].to(self.device)
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views, renderer, self.device, nbv=nbv)
        return render_results, viewmask

    def generate_traj_specified_single(self, c2ws_anchor, H, W, fs, c, theta=0., phi=0., d_r=0., d_x=0., d_y=0., frame=1, device='cuda'):
        """Generate camera trajectory for rendering"""
        c2w_new = sphere2pose(c2ws_anchor, np.float32(theta), np.float32(phi), np.float32(d_r), device, np.float32(d_x), np.float32(d_y))
        c2ws = c2w_new
        num_views = c2ws.shape[0]
        R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
        # Convert from dust3r to pytorch3d coordinate system
        R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2)  # from RDF to LUF for Rotation
        new_c2w = torch.cat([R, T], 2)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)), 1))
        R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3]
        
        image_size = ((H, W),)
        # R_new: (N, 3, 3)
        # T_new: (N, 3)
        # fs: (1)
        # c: (2)
        
        cameras = PerspectiveCameras(
            focal_length=fs,
            principal_point=c,
            in_ndc=False,
            image_size=image_size,
            R=R_new,
            T=T_new,
            device=device
        )
        return cameras, num_views

    def process_single_scene(self, scene_data):
        """Process a single scene using MonST3R and render results"""
        start_time = time.time()
        idx, scene, data_root = scene_data
        
        frame_files = scene['frames'][:5] # test
        frame_paths = [os.path.join(data_root, p) for p in frame_files]
        instance_corners = scene['instance_corners']
            
        save_dir = os.path.join(self.opts.save_dir, f'sample_{idx:06d}')
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 25:
            print(f'skip {save_dir}')
            return None, None
        
        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        # Load and process images using MonST3R pipeline
        images = load_images(frame_paths, size=self.opts.image_size)
        # img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]
        
        scenegraph_type = self.opts.scenegraph_type
        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1
        if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
            scenegraph_type = scenegraph_type + "-" + str(self.opts.swinsize) + "-noncyclic"
        elif scenegraph_type == "oneref":
            scenegraph_type = scenegraph_type + "-" + str(self.opts.refid)

        # Make pairs for reconstruction
        pairs = make_pairs(images, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        
        # Run inference
        output = inference(pairs, self.model, self.device, batch_size=self.opts.batch_size)
        
        # Global alignment
        if len(images) > 2:
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(output, device=self.device, mode=mode)
        else:
            mode = GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=self.device, mode=mode)
        
        # Optimize global alignment
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(
                init='mst',
                niter=self.opts.niter,
                schedule=self.opts.schedule,
                lr=0.01
            )

        # Clean point cloud
        scene = scene.clean_pointcloud()
        
        # Save reconstruction outputs
        save_folder = os.path.join(save_dir, 'reconstruction')
        os.makedirs(save_folder, exist_ok=True)
        
        # Save camera poses and intrinsics
        poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
        K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
        
        # Save depth maps and point clouds
        depth_maps = scene.save_depth_maps(save_folder)
        scene.save_dynamic_masks(save_folder)
        
        c2ws = scene.get_im_poses().detach()
        principal_points = scene.get_principal_points().detach()
        focals = scene.get_focals().detach()
        shape = images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in scene.get_depthmaps()]
        depth_avg = depth[0][H//2,W//2] #以ref图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## masks for cleaner point cloud
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        masks = scene.get_masks()
        depth = scene.get_depthmaps()
        bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks = to_numpy(masks_new)

        ## render, 从c2ws[0]即ref image对应的相机开始
        imgs = np.array(scene.imgs)

        # Transform coordinates to object-centric
        c2ws, pcd = world_point_to_obj(
            poses=c2ws, 
            points=torch.stack(pcd), 
            k=0, # !
            r=radius, 
            elevation=self.opts.elevation, 
            device=self.device
        )
        
        camera_poses = []
        for i in range(len(c2ws)):
            # Generate single transformed pose
            single_pose, _ = self.generate_traj_specified_single(
                c2ws[i:i+1],  # Take one pose at a time
                H, W,
                focals[0:1],  # Use first frame's focal length for all
                principal_points[0:1],  # Use first frame's principal point for all
                self.opts.theta, self.opts.phi, self.opts.d_r, self.opts.d_x, self.opts.d_y,
                1,  # Only generate one frame per pose
                self.device
            )
            camera_poses.append(single_pose)
        
        # Combine all transformed cameras
        camera_traj = camera_poses[0]
        for cam in camera_poses[1:]:
            camera_traj.R = torch.cat([camera_traj.R, cam.R])
            camera_traj.T = torch.cat([camera_traj.T, cam.T])

        print(len(pcd))
        print(len(imgs))
        print(len(c2ws))

        import pdb; pdb.set_trace()

        render_results, _ = self.run_render(
            [pcd[0]],#all_pcd,
            [imgs[0]], #all_imgs, 
            [masks[0]], #masks
            H, W,
            camera_traj,
            len(c2ws),
            self.device
        )

        # Resize renders to target resolution
        render_results = F.interpolate(
            render_results.permute(0,3,1,2),
            size=(576, 1024),
            mode='bilinear',
            align_corners=False
        ).permute(0,2,3,1)

        # render_results[0] = img_ori
        
        # Save results
        save_frames(render_results, save_dir, save_names=[f'render_{idx:06d}_{i:02d}' for i in range(len(render_results))])
        save_video(render_results, os.path.join(save_dir, 'render.mp4'))

        used_time = time.time() - start_time
        return scene, used_time

    def process_dataset(self, json_path, data_root):
        """Process entire dataset using MonST3R"""
        frame_cam_corners_samples = torch.load('../new_frame_cam_corners_samples.pth')
        frame_cam_samples_indices = torch.load('../frame_cam_samples_indices.pth')

        # Filter already processed samples
        rest_frame_cam_corners_samples = []
        rest_frame_cam_samples_indices = []
        for i in range(len(frame_cam_corners_samples)):
            idx = frame_cam_samples_indices[i]
            save_dir = os.path.join(self.opts.save_dir, f'sample_{idx:06d}')
            # if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 25:
            #     print(f'skip {save_dir}')
            #     continue
            rest_frame_cam_corners_samples.append(frame_cam_corners_samples[i])
            rest_frame_cam_samples_indices.append(idx)

        # Get section of data to process
        def split_and_get_nth(data, n, splits=16):
            if n < 0 or n >= splits:
                raise ValueError(f"Invalid split index {n}. Must be between 0 and {splits - 1}.")
            split_size = len(data) // splits
            remainder = len(data) % splits
            splits_indices = []
            start_idx = 0
            for i in range(splits):
                extra = 1 if i < remainder else 0
                splits_indices.append((start_idx, start_idx + split_size + extra))
                start_idx += split_size + extra
            start, end = splits_indices[n]
            return data[start:end]

        split_corners = split_and_get_nth(rest_frame_cam_corners_samples, self.opts.section, self.opts.total_sections)
        split_indices = split_and_get_nth(rest_frame_cam_samples_indices, self.opts.section, self.opts.total_sections)

        # Prepare data for processing
        scene_data = [(idx, scene, data_root) for idx, scene in zip(split_indices, split_corners)]

        # Process scenes
        times = []
        for data in tqdm(scene_data, desc='Processing scenes'):
            # try:
            result, used_time = self.process_single_scene(data)
            if result is not None:
                times.append(used_time)
                avg_time = sum(times) / len(times)
                total_time = avg_time * (len(scene_data) - len(times))
                print(f'Split:{self.opts.section}, time used: {times[-1]:.2f}s, estimated left time: {total_time/3600:.2f}h')
            # except Exception as e:
            #     print(f'Error processing scene: {e}')
            #     continue

if __name__ == "__main__":
    parser = get_parser()
    opts = parser.parse_args()
    
    if opts.exp_name is None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_monst3r'
        
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    os.makedirs(opts.save_dir, exist_ok=True)
    
    monst3r = MonST3R_DataEngine(opts)

    json_path = '/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_frontview_caminex_fullframes.json'
    data_root = "/lpai/dataset/nuscenes-imgs/0-1-0/nuscenes"

    monst3r.process_dataset(json_path, data_root) 