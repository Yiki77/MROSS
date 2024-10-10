import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
import traceback
import numpy as np
import wandb

import config
import sketch_utils as utils
import cv2
def main():
    print("saving iteration video...")
    img_array = []
    for record in range(3):
        for ii in range(0, 2000):
            filename = os.path.join('./output_sketches_3_1-1/scene32/scene32_96strokes_seed0',
                                        "{}_iter_{}.png".format(record, ii))
            img = cv2.imread(filename)
            img_array.append(img)

    videoname = os.path.join('./output_sketches_3_1-1/scene32/scene32_96strokes_seed0',
                             "video.mp4")
    utils.check_and_create_dir(videoname)
    out = cv2.VideoWriter(
        videoname,
        cv2.VideoWriter_fourcc(*'mp4v'),
        # cv2.VideoWriter_fourcc(*'FFV1'),
        10.0, (224, 224))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()
    # print("saving iteration video...")
    # img_array = []
    # for record in range(30):
    #     for ii in range(0, 800):
    #         filename = os.path.join('./output_sketches/17/17_15_15_strokes_seed0',
    #             "{}_iter_{}.png".format(record, ii))
    #         img = cv2.imread(filename)
    #         img_array.append(img)
    #
    # videoname = os.path.join('./output_sketches/17/17_15_15_strokes_seed0',
    #     "video.mp4")
    # utils.check_and_create_dir(videoname)
    # out = cv2.VideoWriter(
    #     videoname,
    #     cv2.VideoWriter_fourcc(*'mp4v'),
    #     # cv2.VideoWriter_fourcc(*'FFV1'),
    #     10.0, (224, 224))
    # for iii in range(len(img_array)):
    #     out.write(img_array[iii])
    # out.release()

if __name__ == "__main__":
    main()