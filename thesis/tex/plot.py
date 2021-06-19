import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math

from pathlib import Path

import click

output = Path("imgs")

# plt.clf()


@click.command()
@click.option("--show", is_flag=True, default=False)
@click.option("--yolo_lr", is_flag=True, default=False)
@click.option("--yolo_offline", is_flag=True, default=False)
@click.option("--yolo_online", is_flag=True, default=False)
@click.option("--yolo_input_tuning", is_flag=True, default=False)
def main(show, yolo_lr, yolo_offline, yolo_online, yolo_input_tuning):
    ########################################################################################
    ## YOLO EXPERIMENTS
    ########################################################################################

    ###########################################
    ## LEARNING RATE SEARCH
    ###########################################
    if yolo_lr:
        lrs = (0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001)
        maps = (72.352, 71.223, 71.474, 73.415, 71.887, 71.472, 70.650)
        width = 0.1
        space = width / 4
        pos = []
        for i in range(len(lrs)):
            pos.append(i * (space + width))


        size = (10, 5)
        fig = plt.figure(figsize=size)

        plt.bar(pos, maps, width=(width for _ in range(len(lrs))))
        plt.xticks(pos, lrs)
        yticks = np.arange(70, 74, 0.25)
        plt.yticks(yticks)
        # plt.set_ylim(ymin=70)
        plt.ylabel("mAP [%]")
        plt.xlabel("Learning Rate")

        # change limit
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 70, 74))


        plt.savefig(str(output / "yolo_lr_experiment.png"))
        if show: plt.show()
        plt.clf()

    ###########################################
    ## OFFLINE AUG
    ###########################################
    if yolo_offline:
        configs = ("Baseline", "R", "F", "F,R", "P", "P,R", "P,F", "P,F,R")
        maps = (73.415, 87.140, 84.415, 91.113, 79.403, 89.641, 86.129, 92.578)

        both = list(zip(configs, maps))
        by_map = lambda vals: vals[1]
        both = sorted(both, key=by_map)

        configs, maps = [], []
        for c, m in both: configs.append(c); maps.append(m);

        width = 0.1
        space = width / 4
        pos = []
        for i in range(len(configs)): pos.append(i *(space + width));

        fig = plt.figure()
        plt.bar(pos, maps, width=(width for _ in range(len(configs))))
        plt.xticks(pos, configs)
        yticks = np.arange(70, 93, 1)
        plt.yticks(yticks)
        plt.ylabel("mAP [%]")
        plt.xlabel("Configuration")

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 70, 93))

        legend = [
            Patch(color="white", label="P: Projection"),
            Patch(color="white", label="F: Horizontal Flip"),
            Patch(color="white", label="R: Rotation")
        ]
        plt.legend(handles=legend)

        plt.savefig(str(output / "yolo_offline_aug_experiment.png"))
        if show: plt.show()
        plt.clf()


    ###########################################
    ## ONLINE AUG
    ###########################################
    if yolo_online:
        rot_map = (95.368, 94.521, 94.198)
        rot_params = (10, 20, 30)

        scale_map = (93.062, 93.261, 92.935)
        scale_params = (0.1, 0.2, 0.3)

        crop_map = (94.820, 94.893, 95.027)
        crop_params = (0.7, 0.8, 0.9)

        color_map = (92.656, 93.243, 93.182)
        color_params = (0.1, 0.2, 0.3)

        maps = np.array([
            rot_map, scale_map, crop_map, color_map
        ]).T







    ###########################################
    ## TUNING INPUT SIZE
    ###########################################
    if yolo_input_tuning:
        input_size = (544, 576, 608, 640, 672, 704, 736, 768, 800, 832)
        valid = (92.534, 95.237, 96.370, 96.210, 95.659, 96.541, 97.006, 96.571, 96.359, 95.791)
        test = (86.674, 87.695, 88.884, 91.531, 91.871, 92.227, 92.925, 93.506, 92.890, 92.885)

        size = (3, 3)
        fig = plt.figure(figsize=size)
        both = [*valid, *test]
        plt.xticks(input_size)
        plt.yticks(range(int(min(both)), math.ceil(max(both))))

        plt.plot(input_size, valid, "-o")
        plt.plot(input_size, test, "-o")

        plt.legend(("Validation", "Test"))

        if show:
            plt.show()
        plt.savefig(output / "yolo_input_size_tuning.png")
        plt.clf()


    ########################################################################################
    ## MUNET EXPERIMENTS
    ########################################################################################


    ########################################################################################
    ## tuning unet input size
    ########################################################################################

if __name__ == "__main__":
    main()
