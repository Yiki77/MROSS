#MROSS
#python run_scene_sketching.py --target scene65.png --region_round 1 --num_strokes 48  --num_sketches 1
import warnings
import CLIP_.clip as clip
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import sys
import time
import traceback
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange
import dlib
import config
import sketch_utils as utils
from models.loss import Loss, LPIPS
from models.painter_params import XDoG_, Painter, PainterOptimizer, interpret
from IPython.display import display, SVG
from torchvision.utils import save_image
import cv2
import random
from collections import OrderedDict
def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_strokes, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_

def get_roi(args):
    roi_ = []
    for round in range (args.region_round - 1):
        roi = f"./2-target_images/{args.target_file[:-4]}_roi{round+1}{args.target_file[-4:]}"
        print("roi_name===",roi)
        roi = Image.open(roi)
        if roi.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", roi.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(roi, (0, 0), roi)
            roi = new_image
        roi = roi.convert("RGB")

        if args.fix_scale:
            target = utils.fix_image_scale(roi)

        transforms_ = []
        if roi.size[0] != roi.size[1]:
            transforms_.append(transforms.Resize(
                (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
        else:
            transforms_.append(transforms.Resize(
                args.image_scale, interpolation=PIL.Image.BICUBIC))
            transforms_.append(transforms.CenterCrop(args.image_scale))
        transforms_.append(transforms.ToTensor())
        data_transforms = transforms.Compose(transforms_)
        roi_.append(data_transforms(roi).unsqueeze(0).to(args.device))
    return roi_

def fps(points, num_points):
    """
    FPS algorithm
    Args:
        points: point sets，N x 2 2D array，each line represent an axis of point
        num_points: number of sampling points
    Returns:
        list for sampled points
    """
    n = len(points)
    distances = np.full(n, np.inf)  
    # distances = 0  
    samples = []  
    samples_points = []  
    current = np.random.randint(n)  
    samples.append(current) 
    samples_points.append(points[current+10]) 

    while len(samples) < num_points:
        for i in range(n):
            if i in samples:
                distances[i] = 0
            else:
                dist = np.linalg.norm(points[i] - points[current])
                distances[i] = min(dist, distances[i])
        farthest = np.argmax(distances)
        samples.append(farthest)
        samples_points.append(points[farthest])
        current = farthest
        distances[current] = np.inf 

    return np.array(samples_points)

def get_edge_points(image):
    image = image.detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    # Using Canny detector
    edges = cv2.Canny(blur, 20, 200)
    edge_points = np.argwhere(edges > 0)  # Dectect points in edge
    if edge_points.size == 0:
        return []
    cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
    return edge_points


def extract_global_points(image_path, num_points):
    # Read image and transfer to gray
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    # image = image.detach().cpu().numpy()
    # image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    # Using Canny detector
    edges = cv2.Canny(blur, 20, 200)
    edge_points = np.argwhere(edges > 0)  # Dectect points in edge
    if edge_points.size == 0:
        return []
    cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
    if num_points == 0:
        selected_points = []
    else:
        selected_points = fps(edge_points, num_points)
    selected_points = np.array(selected_points)
    print("selected_points2",selected_points)
    return selected_points

def extract_region_points(args, num_points):
    region_points = []
    # Read image and transfer to gray
    for round in range (args.region_round - 1):
        roi_path = f"./2-target_images/{args.target_file[:-4]}_roi{round+1}{args.target_file[-4:]}"
        image = cv2.imread(roi_path)
        image = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (3, 3))
        # Using Canny detector
        edges = cv2.Canny(blur, 20, 200)
        edge_points = np.argwhere(edges > 0)  # Detect points in edge
        if edge_points.size == 0:
            return []
        cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
        if num_points[round] == 0:
            selected_points = []
        else:
            selected_points = fps(edge_points, num_points[round])
        region_points.append(np.array(selected_points))
    print("region_points",region_points)
    return region_points

def get_edge_pathnum(args):
    # Read image and transfer to gray
    target = cv2.imread(args.target)
    target = cv2.resize(target, (224, 224))
    # image = image.detach().cpu().numpy()
    # image = (image * 255).astype(np.uint8)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target_blur = cv2.blur(target_gray, (3, 3))
    # Using Canny detector
    target_edges = cv2.Canny(target_blur, 20, 200)
    target_edge_points = np.argwhere(target_edges > 0)  # Detect points in edge

    cv2.imwrite("{}/target_edge_image.png".format(args.output_dir), target_edges)
    len_target = target_edge_points.size
    print("Number of target edge:", len_target)

    len_roi= []
    roi_pathnum= []
    for i in range(args.region_round - 1):
        roi = f"./2-target_images/{args.target_file[:-4]}_roi{i+1}{args.target_file[-4:]}"
        # Read image and transfer to gray
        roi = cv2.imread(roi)
        roi = cv2.resize(roi, (224, 224))
        # image = image.detach().cpu().numpy()
        # image = (image * 255).astype(np.uint8)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.blur(roi_gray, (3, 3))
        # Using Canny detector
        roi_edges = cv2.Canny(roi_blur, 20, 200)
        roi_edge_points = np.argwhere(roi_edges > 0)  # Detect points in edge
        cv2.imwrite("{}/roi{}_edge_image.png".format(args.output_dir, i), roi_edges)
        len_roi.append(roi_edge_points.size)

    target_roi = len_target / (len_target + sum(len_roi))
    target_pathnum = round(target_roi * args.num_strokes)
    for i in range(args.region_round - 1):
        roi_roi = len_roi[i] / (len_target + sum(len_roi))
        roi_pathnum.append(round(roi_roi * args.num_strokes))

    pathsum = target_pathnum + sum(roi_pathnum)
    #Adjust stroke number
    if pathsum > args.num_strokes:
        target_pathnum = target_pathnum - (pathsum - args.num_strokes)
    return target_pathnum, roi_pathnum

def get_pixel_pathnum(args):
    # Load image
    target = cv2.imread(args.target)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target_thresh = cv2.threshold(target_gray, 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"{}/target_thresh.png".format(args.output_dir), cv2.bitwise_not(target_thresh))
    target_pixels = cv2.countNonZero(cv2.bitwise_not(target_thresh))
    print("Number of target pixels:", target_pixels)

    roi_pixels= []
    roi_pathnum= []
    for i in range(args.region_round - 1):
        roi = f"./2-target_images/{args.target_file[:-4]}_roi{i+1}{args.target_file[-4:]}"
        roi_round = cv2.imread(roi)
        roi_gray = cv2.cvtColor(roi_round, cv2.COLOR_BGR2GRAY)

        roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(r"{}/{}_roi_thresh.png".format(round, args.output_dir), cv2.bitwise_not(roi_thresh))

        roi_pixels.append(cv2.countNonZero(cv2.bitwise_not(roi_thresh)))

    target_roi = target_pixels / (target_pixels + sum(roi_pixels))
    target_pathnum = round(target_roi * args.num_strokes)
    for i in range(args.region_round - 1):
        roi_roi = roi_pixels[i] / (target_pixels + sum(roi_pixels))
        roi_pathnum.append(round(roi_roi * args.num_strokes))

    pathsum = target_pathnum + sum(roi_pathnum)
    #Adjust suitable number
    if pathsum > args.num_strokes:
        target_pathnum = target_pathnum - (pathsum - args.num_strokes)
    return target_pathnum, roi_pathnum
def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

def get_new_points(args, target_im):
    model, preprocess = clip.load(args.saliency_clip_model, device=args.device, jit=False)
    model.eval().to(args.device)

    data_transforms = transforms.Compose([
        preprocess.transforms[-1],
    ])
    image_input_attn_clip = data_transforms(target_im).to(args.device)

    attention_map = interpret(image_input_attn_clip, model, device=args.device)
    del model
    attn_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min())
    if args.xdog_intersec:
        xdog = XDoG_()
        im_xdog = xdog(image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=15)
        intersec_map = (1 - im_xdog) * attn_map
        attn_map = intersec_map
    cv2.imwrite(r"{}/attn_map.png".format(args.output_dir), attn_map)

    attn_map_soft = np.copy(attn_map)
    attn_map_soft[attn_map > 0] = softmax(attn_map[attn_map > 0], tau=args.softmax_temp)

    # k = int((self.num_stages * self.num_paths)/self.masknum)
    numsize = 1
    k = args.num_stages * numsize

    inds = np.random.choice(range(attn_map.flatten().shape[0]), size=1, replace=False,
                                     p=np.max(attn_map_soft.flatten()))
    inds = np.array(np.unravel_index(inds, attn_map.shape)).T

    inds_normalised_record = np.zeros(inds.shape)
    inds_normalised_record[:, 0] = inds[:, 1] / args.image_scale
    inds_normalised_record[:, 1] = inds[:, 0] / args.image_scale
    inds_normalised_record = inds_normalised_record.tolist()
    return inds_normalised_record

def random_points(image_path, num_points):
    image = Image.open(image_path)

    # width, height = image.size
    region_points = []
    # New image, draw points
    points_image = Image.new('RGB', (224, 224))

    # radom axis and select x_points, y_points
    for _ in range(num_points):
        x = random.randint(0, 224 - 1)
        y = random.randint(0, 224 - 1)
        region_points.append([x,y])

    return region_points

def main(args):
    loss_func = Loss(args)
    loss_lpips = LPIPS(args).to(args.device)
    inputs_ = get_target(args)
    # roi_ = get_roi(args)
    inputs = inputs_
    utils.log_input(args.use_wandb, 0, inputs_, args.output_dir)
    target_pathnum, roi_pathnum = get_edge_pathnum(args)
    # print("target_pathnum===",target_pathnum)
    # print("roi_pathnum===",roi_pathnum)
    renderer = load_renderer(args, inputs)
    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    if args.num_hair_paths != 0 :
        # Attention Sample
        # global_inds = renderer.get_attn_global_points()
        #FPS Sample
        global_inds = extract_global_points(args.target, target_pathnum)
        #Random Sample
        # global_inds = random_points(args.target, target_pathnum)
        global_inds = np.array(global_inds)
    else :
        global_inds = [[ ]]
    if args.num_face_paths != 0:
        region_inds = extract_region_points(args,roi_pathnum)
    else :
        region_inds = [[]]
   
    #Iteration
    inds_num = 0
    for record in range(args.region_round):
        if record == 0:
            inds_record = global_inds
            num_paths = target_pathnum
            if num_paths == 0:
                continue
        else:
            inds_record = region_inds[record-1]
            num_paths = roi_pathnum[record-1]
            if num_paths == 0:
                continue


        inds_num += num_paths
        inds_normalised_record = np.zeros(inds_record.shape)
        inds_normalised_record[:, 0] = inds_record[:, 1] / args.image_scale
        inds_normalised_record[:, 1] = inds_record[:, 0] / args.image_scale
        inds_normalised_record = inds_normalised_record.tolist()


        renderer.set_random_noise(0)
        renderer.init_image_1(inds_normalised_record, num_paths)
        optimizer.init_optimizers()

        # not using tdqm for jupyter demo
        if args.display:
            epoch_range = range(args.num_iter)
        else:
            epoch_range = tqdm(range(args.num_iter))

        #Optimization
        for epoch in epoch_range:
            if not args.display:
                epoch_range.refresh()
            renderer.set_random_noise(epoch)
            if args.lr_scheduler:
                optimizer.update_lr(counter)

            start = time.time()
            optimizer.zero_grad_()
            sketches = renderer.get_image().to(args.device)

           # Loss
            losses_dict = loss_func(sketches, inputs.detach(
            ), renderer.get_color_parameters(), renderer, counter, optimizer)
            loss_clip = sum(list(losses_dict.values()))
            loss_vgg = torch.mean(loss_lpips(sketches, inputs.detach()))
            
            loss = 1 * loss_clip + 1 * loss_vgg
            print("final loss == ", loss)
            loss.backward()
            optimizer.step_()
            if epoch % args.save_interval == 0:
                save_image(sketches, "{}/{}_iter_{}.png".format(args.output_dir, record, epoch))
                renderer.save_svg(
                    f"{args.output_dir}/svg_logs", f"{record}_svg_iter{epoch}")

            utils.plot_target_clip(inputs, np.array(global_inds), "{}/{}_{}.jpg".format(
                    args.output_dir, "target_points_map", target_pathnum))
            if record !=0 :
                utils.plot_roi_clip(inputs, np.array(region_inds[record-1]), "{}/roi_{}_{}_{}.jpg".format(
                        args.output_dir, record, "points_map", roi_pathnum[record-1]))

            if args.use_wandb:
                wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
                for k in losses_dict.keys():
                    wandb_dict[k] = losses_dict[k].item()
                wandb.log(wandb_dict, step=counter)

            counter += 1

    #Video
    if args.save_video:
        print("saving iteration video...")
        img_array = []
        for ii in range(0, args.num_iter):
            filename = os.path.join(
                "{}/".format(args.output_dir),
                "iter_{}.png".format(ii))
            img = cv2.imread(filename)
            cv2.putText(
                img, "global_path:{} \nface_path:{}".format(args.num_global_paths, args.num_region_paths),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            img_array.append(img)
    
        videoname = os.path.join(
            args.output_dir,"iter"
            "h{}_f{}.avi".format(args.num_global_paths,args.num_region_paths))
        utils.check_and_create_dir(videoname)
        out = cv2.VideoWriter(
            videoname,
            # cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'FFV1'),
            10.0, (args.image_scale, args.image_scale))
        for iii in range(len(img_array)):
            out.write(img_array[iii])
        out.release()
    print("Final Number of target pixels:", target_pathnum)
    print("Final Number of roi pixels:", roi_pathnum)
    renderer.save_svg(args.output_dir, "final_svg")
    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()