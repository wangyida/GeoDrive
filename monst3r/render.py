# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
from fileinput import fileno
import math
from struct import calcsize
from altair import Theta
import gradio
import os
from sympy import ask
import torch
import numpy as np
import tempfile
import functools
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
import matplotlib.pyplot as pl
import cv2

from pvd_utils import world_point_to_obj, sphere2pose, generate_traj_specified, setup_renderer, save_video, save_frames

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

## render imports
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Pointclouds
from PIL import Image
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import time

def render_pcd(pts3d,imgs,masks,views,renderer,device='cpu',nbv=False):
    # Placeholder for rendering point cloud
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)

    if masks == None:
        pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
    else:
        # masks = to_numpy(masks)
        pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
        col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
    
    point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
    images = renderer(point_cloud)

    if nbv:
        color_mask = torch.ones(col.shape).to(device)
        point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
        view_masks = renderer(point_cloud_mask)
    else: 
        view_masks = None

    return images, view_masks

def run_render(pcd, imgs,masks, H, W, camera_traj,num_views,device='cpu', nbv=False):
    # Placeholder for running the rendering process
    render_setup = setup_renderer(camera_traj, image_size=(H,W))
    renderer = render_setup['renderer']
    render_results, viewmask = render_pcd(pcd, imgs, masks, num_views,renderer,device=device,nbv=nbv)
    return render_results, viewmask

def generate_traj_from_single_c2w(c2ws_anchor,H,W,fs,c,theta=0., phi=0.,d_r=0.,d_x=0.,d_y=0.,frame=1,device='cuda'):
    # Placeholder for generating trajectory from a single camera to world transformation
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


def generate_traj_from_c2ws(c2ws_anchor,H,W,fs,c,frame,device):
    # Placeholder for generating trajectory from multiple camera to world transformations
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    theta = 0
    phi = 0
    r = 1
    x = 0
    y = 0

    c2ws_list = []
    for i in range(frame):
        c2w_new = sphere2pose(c2ws_anchor[i:i+1], np.float32(theta), np.float32(phi), np.float32(r), device, np.float32(x),np.float32(y))
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
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


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--weights", type=str, help="path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--not_batchify', action='store_true', default=False, help='Use non batchify mode for global optimization')
    parser.add_argument('--real_time', action='store_true', default=False, help='Realtime mode')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')

    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')
    
    # render options
    parser.add_argument('--total_sections', type=int, default=1, help='Total sections for rendering')
    parser.add_argument('--section', type=int, default=0, help='Section number for rendering')
    
    return parser

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    # Placeholder for getting 3D model from scene
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]

    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)

def render_2D_images_from_scene(outdir, scene, min_conf_thr=3, mask_sky=False,
                            clean_depth=False, thr_for_init_conf=True):
    # Placeholder for rendering 2D images from the scene
    """
    render 2D images from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    # scene.min_conf_thr = min_conf_thr
    # scene.thr_for_init_conf = thr_for_init_conf
    # msk = to_numpy(scene.get_masks())
    msk = None
   
    # DEBUG, render images
    elevation = 5.0
    center_scale = 1.0
    d_x = torch.tensor(0.0)
    d_y = torch.tensor(0.0)
    d_r = torch.tensor(0.0)
    d_theta = torch.tensor(0.0)
    d_phi = torch.tensor(0.0)
    dpt_trd = torch.tensor(1.0)
    device = torch.device('cuda')
    H, W = rgbimg[0].shape[:2]

    c2ws = scene.get_im_poses().detach()
    pcd = [i.detach() for i in scene.get_pts3d(clip_thred=dpt_trd, raw_pts=True)] # a list of points of size whc
    focals = scene.get_focals().detach()
    principal_points = scene.get_principal_points().detach()
    depth = to_numpy(scene.get_depthmaps())
    depth_avg = depth[0][H//2,W//2]
    radius = depth_avg*center_scale 

    c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=radius, elevation=elevation, device=device)

    # Apply the fixed view transformation to all camera poses
    camera_poses = []
    for i in range(len(c2ws)):
        # Generate single transformed pose
        single_pose, _ = generate_traj_from_single_c2w(
            c2ws[i:i+1],  # Take one pose at a time
            H, W,
            focals[0:1],  # Use first frame's focal length for all
            principal_points[0:1],  # Use first frame's principal point for all
            frame=1,  # Only generate one frame per pose
            device=device
        )
        camera_poses.append(single_pose)

    # Combine all transformed cameras
    camera_traj = camera_poses[0]
    for cam in camera_poses[1:]:
        camera_traj.R = torch.cat([camera_traj.R, cam.R])
        camera_traj.T = torch.cat([camera_traj.T, cam.T])
    num_views = camera_traj.R.shape[0]
    
    # use first frame point cloud and image
    render_results, viewmask = run_render(pcd[0:1], rgbimg[0:1], msk, H, W, camera_traj,num_views, device=device, nbv=False)

    # put ref image into the first position
    render_results[0] = torch.from_numpy(rgbimg[0])

    render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
    save_frames(render_results, outdir, save_names=[f'render_{i}' for i in range(len(render_results))], folder=outdir)
    save_video(render_results, os.path.join(outdir, 'render.mp4'))

    return render_results

def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, num_frames):
    # Placeholder for getting reconstructed scene
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72, batchify=not args.not_batchify)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
   
    render_results = render_2D_images_from_scene(save_folder, scene, min_conf_thr, mask_sky,
                            clean_depth, thr_for_init_conf=True)    

    # save_folder = f'{save_folder}/reconstruction'  #default is 'demo_tmp/NULL'
    # poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    # K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    # depth_maps = scene.save_depth_maps(save_folder)
    # dynamic_masks = scene.save_dynamic_masks(save_folder)
    # conf = scene.save_conf_maps(save_folder)
    # init_conf = scene.save_init_conf_maps(save_folder)
    # rgbs = scene.save_rgb_imgs(save_folder)
    # enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 

    # # also return rgb, depth and confidence imgs
    # # depth is normalized with the max value for all images
    # # we apply the jet colormap on the confidence maps
    # rgbimg = scene.imgs
    # depths = to_numpy(scene.get_depthmaps())
    # confs = to_numpy([c for c in scene.im_conf])
    # init_confs = to_numpy([c for c in scene.init_conf_maps])
    # cmap = pl.get_cmap('jet')
    # depths_max = max([d.max() for d in depths])
    # depths = [cmap(d/depths_max) for d in depths]
    # confs_max = max([d.max() for d in confs])
    # confs = [cmap(d/confs_max) for d in confs]
    # init_confs_max = max([d.max() for d in init_confs])
    # init_confs = [cmap(d/init_confs_max) for d in init_confs]

    # imgs = []
    # for i in range(len(rgbimg)):
    #     imgs.append(rgbimg[i])
    #     imgs.append(rgb(depths[i]))
    #     imgs.append(rgb(confs[i]))
    #     imgs.append(rgb(init_confs[i]))

    # # if two images, and the shape is same, we can compute the dynamic mask
    # if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
    #     motion_mask_thre = 0.35
    #     error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir, motion_mask_thre=motion_mask_thre)
    #     # imgs.append(rgb(error_map))
    #     # apply threshold on the error map
    #     normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    #     error_map_max = normalized_error_map.max()
    #     error_map = cmap(normalized_error_map/error_map_max)
    #     imgs.append(rgb(error_map))
    #     binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
    #     imgs.append(rgb(binary_error_map*255))

    # used_time = time.time() - start_time

    del scene
    torch.cuda.empty_cache()

    return render_results

def get_reconstructed_scene_realtime(args, model, device, silent, image_size, filelist, scenegraph_type, refid, seq_name, fps, num_frames):
    # Placeholder for getting reconstructed scene in real-time
    model.eval()
    imgs = load_images(filelist, size=image_size, verbose=not silent, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    
    if scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    elif scenegraph_type == "oneref_mid":
        scenegraph_type = "oneref-" + str(len(imgs) // 2)
    else:
        raise ValueError(f"Unknown scenegraph type for realtime mode: {scenegraph_type}")
    
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)


    view1, view2, pred1, pred2 = output['view1'], output['view2'], output['pred1'], output['pred2']
    pts1 = pred1['pts3d'].detach().cpu().numpy()
    pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
    for batch_idx in range(len(view1['img'])):
        colors1 = rgb(view1['img'][batch_idx])
        colors2 = rgb(view2['img'][batch_idx])
        xyzrgb1 = np.concatenate([pts1[batch_idx], colors1], axis=-1)   #(H, W, 6)
        xyzrgb2 = np.concatenate([pts2[batch_idx], colors2], axis=-1)
        np.save(save_folder + '/pts3d1_p' + str(batch_idx) + '.npy', xyzrgb1)
        np.save(save_folder + '/pts3d2_p' + str(batch_idx) + '.npy', xyzrgb2)

        conf1 = pred1['conf'][batch_idx].detach().cpu().numpy()
        conf2 = pred2['conf'][batch_idx].detach().cpu().numpy()
        np.save(save_folder + '/conf1_p' + str(batch_idx) + '.npy', conf1)
        np.save(save_folder + '/conf2_p' + str(batch_idx) + '.npy', conf2)

        # save the imgs of two views
        img1_rgb = cv2.cvtColor(colors1 * 255, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(colors2 * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_folder + '/img1_p' + str(batch_idx) + '.png', img1_rgb)
        cv2.imwrite(save_folder + '/img2_p' + str(batch_idx) + '.png', img2_rgb)

    return save_folder


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    
    json_path = '/lpai/volumes/ad-vla-vol-ga/chenantong/datasets/NuScenes/annotation/nuScenes_frontview_caminex_fullframes.json'
    data_root = "/lpai/dataset/nuscenes-imgs/0-1-0/nuscenes"
    frame_cam_corners_samples = torch.load('../new_frame_cam_corners_samples.pth')
    frame_cam_samples_indices = torch.load('../frame_cam_samples_indices.pth')

    # Filter already processed samples
    rest_frame_cam_corners_samples = []
    rest_frame_cam_samples_indices = []
    for i in range(len(frame_cam_corners_samples)):
        idx = frame_cam_samples_indices[i]
        save_dir = os.path.join(args.output_dir, f'sample_{idx:06d}')
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 25:
            print(f'skip {save_dir}')
            continue
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

    split_corners = split_and_get_nth(rest_frame_cam_corners_samples, args.section, args.total_sections)
    split_indices = split_and_get_nth(rest_frame_cam_samples_indices, args.section, args.total_sections)

    # Prepare data for processing
    scene_data = [(idx, scene, data_root) for idx, scene in zip(split_indices, split_corners)]

    # Process scenes
    times = []
    for data in tqdm(scene_data, desc='Processing scenes'):
        
        # result, used_time = self.process_single_scene(data)
        
        sample_idx, scene, data_root = data
        frame_files = scene['frames']
        assert len(frame_files) == 25
        frame_paths = [os.path.join(data_root, p) for p in frame_files]
        instance_corners = scene['instance_corners']

        if len(instance_corners) > 0:
            print('skip dynamic scene for now...')
            continue

        start = time.time()
        results = get_reconstructed_scene(
            args, args.output_dir, model, args.device, args.silent, args.image_size,
            filelist=frame_paths,
            schedule='linear',
            niter=300,
            min_conf_thr=1.1,
            as_pointcloud=True,
            mask_sky=False,
            clean_depth=True,
            transparent_cams=False,
            cam_size=0.05,
            show_cam=True,
            scenegraph_type='swin', # swin, swinstride, swin2stride, oneref
            winsize=3,
            refid=0,
            seq_name=f'sample_{sample_idx:06d}',
            new_model_weights=args.weights,
            temporal_smoothing_weight=0.01,
            translation_weight='1.0',
            shared_focal=True,
            flow_loss_weight=0.01,
            flow_loss_start_iter=0.1,
            flow_loss_threshold=25,
            use_gt_mask=args.use_gt_davis_masks,
            fps=args.fps,
            num_frames=args.num_frames,
        )
        used_time = time.time() - start
        print(f"Processing completed. Output saved in {save_dir}/{args.seq_name}")

        if results is not None:
            times.append(used_time)
            avg_time = sum(times) / len(times)
            total_time = avg_time * (len(scene_data) - len(times))
            print(f'Split:{args.section}, time used: {times[-1]:.2f}s, estimated left time: {total_time/3600:.2f}h')