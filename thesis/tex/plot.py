import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import math
import itertools as it

from sklearn.metrics import precision_recall_curve

import seaborn as sns
import pandas as pd
from pathlib import Path

import click

output = Path("imgs")

# plt.clf()

gamma = "\u03b3"
alpha = "\u03b1"


def fsci(f, *p):
    if f == 0.01:
        return "1.0e-2"
    if f == 0.005:
        return "5.0e-3"
    if f == 0.0025:
        return "2.5e-3"
    if f == 0.001:
        return "1.0e-3"
    if f == 0.0005:
        return "5.0e-4"
    if f == 0.00025:
        return "2.5e-4"
    if f == 0.0001:
        return "1.0e-4"

    return None


green = sns.color_palette("Greens_d", 3)[1]
blue = sns.color_palette("Blues_d", 3)[1]
red = sns.color_palette("Reds_d", 3)[1]


def text_on_bars(ax, values, formatter=lambda v: f"{v:.3f}%", fontsize=None):
    bars = ax.patches
    print(len(bars))
    for bar, value in zip(bars, values):
        h = bar.get_height()
        w = bar.get_width()
        pos = bar.get_x() + bar.get_width() / 2.0

        if fontsize:
            ax.text(pos, h, formatter(value), ha="center", fontsize=fontsize)
        else:
            ax.text(pos, h, formatter(value), ha="center")


def set_capcolor(ax, color=green):
    for line in ax.get_lines():
        line.set_color(color)


def make_score_tuning_heatmap(results, score_threshs, iou_threshs, size=(11, 11)):
    ylabel, xlabel = "IoU Threshold", "Score Threshold"

    fig = plt.figure(figsize=size)
    ax = sns.heatmap(results[::-1], annot=True, fmt=".5f", square=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(score_threshs)
    ax.set_yticklabels(iou_threshs[::-1])
    ax.tick_params(axis="y", labelrotation=25)
    plt.tight_layout()


error_color = (0, 0, 0, 0.3)

# fmt: off
@click.command()
@click.option("--show",                 is_flag=True, default=False)
@click.option("--pr_curve",             is_flag=True, default=False)
@click.option("--yolo_lr",              is_flag=True, default=False)
@click.option("--yolo_offline",         is_flag=True, default=False)
@click.option("--yolo_online",          is_flag=True, default=False)
@click.option("--yolo_grid_all",        is_flag=True, default=False)
@click.option("--yolo_grid_heat",       is_flag=True, default=False)
@click.option("--yolo_input_size",      is_flag=True, default=False)
@click.option("--yolo_diou_heat",       is_flag=True, default=False)
@click.option("--yolo_diou_tta_heat",   is_flag=True, default=False)
@click.option("--yolo_wbf_heat",        is_flag=True, default=False)
@click.option("--yolo_wbf_tta_heat",    is_flag=True, default=False)
@click.option("--yolo_wbf_tta_votes",   is_flag=True, default=False)
@click.option("--yolo_all_tuning",      is_flag=True, default=False)
@click.option("--munet_lr",             is_flag=True, default=False)
@click.option("--munet_offline",        is_flag=True, default=False)
@click.option("--munet_online",         is_flag=True, default=False)
@click.option("--munet_grid_all_bs",    is_flag=True, default=False)
@click.option("--munet_grid_all",       is_flag=True, default=False)
@click.option("--munet_grid_heat",      is_flag=True, default=False)
@click.option("--munet_folds",          is_flag=True, default=False)
def main(show, pr_curve, yolo_lr, yolo_offline, yolo_online, yolo_grid_all, yolo_grid_heat, yolo_input_size, yolo_diou_heat, yolo_wbf_heat, yolo_diou_tta_heat, yolo_wbf_tta_heat, yolo_wbf_tta_votes, yolo_all_tuning, munet_lr, munet_offline, munet_online, munet_grid_all_bs, munet_grid_all, munet_grid_heat, munet_folds):
# fmt: on
    if pr_curve:
        sns.set_style("whitegrid")
        ylabel, xlabel = "Precision", "Recall"

        data = pd.DataFrame({
            xlabel: np.arange(0, 1.1, 0.1),
            ylabel: (1, 1, 1, 0.4, 0.45, 0.5, 0.55, 0.58, 0.45, 0.48, 0.5)
        })

        # ax = sns.lineplot(data=data, x=xlabel, y=ylabel)
        # ax.set_yticks(np.arange(0, 1.1, 0.1))
        # ax.set_xticks(np.arange(0, 1.1, 0.1))

        plt.savefig(output / "pr_curve.pdf")
        if show:
            plt.show()

    ########################################################################################
    ## YOLO EXPERIMENTS
    ########################################################################################

    ###########################################
    ## LEARNING RATE SEARCH
    ###########################################
    if yolo_lr:
        sns.set_style("whitegrid")
        lrs = [fsci(lr) for lr in (0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001)]
        map_values = (72.352, 71.223, 71.474, 73.415, 71.887, 71.472, 70.650)
        maps = (
            (0.730395, 0.722359, 0.717805),
            (0.731512, 0.712088, 0.693098),
            (0.726695, 0.733718, 0.683804),
            (0.734765, 0.725393, 0.742283),
            (0.727844, 0.722296, 0.70647),
            (0.71887, 0.714032, 0.711261),
            (0.700158, 0.722826, 0.69653),
        )

        ylabel, xlabel = "mAP [%]", "Learning Rate"

        data = []
        for lr, map_ in zip(lrs, maps):
            for v in map_:
                data.append((lr, v * 100))

        data = pd.DataFrame(data, columns=[xlabel, ylabel])

        # size = (8, 5)
        size = (22, 10)

        fig = plt.figure(figsize=size)
        plt.ylim((67, 75))
        ax = sns.barplot(
            x=xlabel,
            y=ylabel,
            data=data,
            palette=len(lrs) * [blue],
            capsize=0.2,
            ci="sd",
            errcolor=error_color,
        )
        ax.set_ylabel(ylabel, fontsize=34)
        ax.set_xlabel(xlabel, fontsize=34)
        ax.tick_params(labelsize=28)
        sns.set_style("whitegrid")
        text_on_bars(ax, map_values, lambda v: f"{v:.3f}%", fontsize=28)


        plt.savefig(output / "yolo_lr_experiment.pdf")
        if show:
            plt.show()

    ###########################################
    ## OFFLINE AUG
    ###########################################
    if yolo_offline:
        sns.set_style("whitegrid")
        configs = ("Baseline", "R", "F", "F,R", "P", "P,R", "P,F", "P,F,R")
        map_vals = (73.415, 87.140, 84.415, 91.113, 79.403, 89.641, 86.129, 92.578)

        maps = (
            (0.734765, 0.725393, 0.742283),
            (0.874025, 0.873762, 0.866413),
            (0.884209, 0.825059, 0.82317),
            (0.914477, 0.910632, 0.90827),
            (0.817009, 0.776369, 0.788722),
            (0.897369, 0.896583, 0.895264),
            (0.861265, 0.859016, 0.863599),
            (0.924946, 0.930219, 0.92216),
        )

        all_ = list(zip(configs, map_vals, maps))
        by_map_mean = lambda vals: vals[1]
        all_ = sorted(all_, key=by_map_mean)

        xlabel, ylabel = "Configuration", "mAP [%]"

        data = []
        for config, mean, maps in all_:
            for v in maps:
                data.append((config, v * 100))

        data = pd.DataFrame(data, columns=[xlabel, ylabel])

        # size = (9, 6)
        size = (22, 13)
        fig = plt.figure(figsize=size)
        min_y, max_y, step = 68, 94, 2
        ax = sns.barplot(
            x=xlabel,
            y=ylabel,
            data=data,
            palette=[green] + (len(configs) - 1) * [blue],
            capsize=0.2,
            errcolor=error_color,
            ci="sd",
        )
        ax.set_ylabel(ylabel, fontsize=34)
        ax.set_xlabel(xlabel, fontsize=34)
        ax.tick_params(labelsize=28)
        text_on_bars(ax, [v for _, v, _ in all_], lambda v: f"{v:.3f}%", fontsize=28)

        # ticks
        yticks = np.arange(min_y, max_y, step)
        ax.set_yticks(yticks)
        plt.ylim((min_y, max_y))

        plt.rcParams["legend.fontsize"] = 28
        legend = [
            Patch(color="white", label="P: Projection"),
            Patch(color="white", label="F: Horizontal Flip"),
            Patch(color="white", label="R: Rotation"),
            Patch(color="white", label="Baseline: LR = 1e-3"),
        ]
        plt.legend(handles=legend, loc="upper left")

        plt.savefig(output / "yolo_offline_aug_experiment.pdf")
        if show:
            plt.show()

    ###########################################
    ## ONLINE AUG
    ###########################################
    if yolo_online:
        augs = ("Rotation", "Scale", "SafeCrop", "ColorJitter")

        rot_map_vals = (95.368, 94.521, 94.198)
        rot_params = ("10°", "20°", "30°")
        rot_maps = (
            (0.957676, 0.953432, 0.949919),
            (0.945191, 0.94476, 0.94568),
            (0.940533, 0.947177, 0.938232),
        )

        scale_map_vals = (93.062, 93.261, 92.935)
        scale_params = ("10%", "20%", "30%")
        scale_maps = (
            (0.92964, 0.928164, 0.934063),
            (0.924235, 0.938415, 0.935191),
            (0.936594, 0.926332, 0.925122),
        )

        crop_map_vals = (94.820, 94.893, 95.027)
        crop_params = ("70%", "80%", "90%")
        crop_maps = (
            (0.947463, 0.947819, 0.949305),
            (0.948257, 0.945725, 0.952799),
            (0.949988, 0.945555, 0.955263),
        )

        color_map_vals = (92.656, 93.243, 93.182)
        # hack to prevent the combination of the same xvalues
        # this is an invisible unicode char
        color_params = ("10%\uFEFF", "20%\uFEFF", "30%\uFEFF")
        color_maps = (
            (0.923355, 0.930764, 0.925557),
            (0.930736, 0.936396, 0.930167),
            (0.94089, 0.927393, 0.92719),
        )

        baseline_map_val = (92.578,)
        baseline_maps = ((0.924946, 0.930219, 0.92216),)

        baseline_params = ("Baseline",)

        xlabel, ylabel = "Parameters", "mAP [%]"
        sns.set(font_scale=2)
        sns.set_style("whitegrid")

        data = []

        def to_data(aug, param, maps):
            for p, maps_ in zip(param, maps):
                for v in maps_:
                    data.append((aug, p, v * 100))

        to_data("Baseline", baseline_params, baseline_maps)
        to_data("Rotation", rot_params, rot_maps)
        to_data("Scale", scale_params, scale_maps)
        to_data("SafeCrop", crop_params, crop_maps)
        to_data("ColorJitter", color_params, color_maps)
        data = pd.DataFrame(data, columns=["Augmentation", xlabel, ylabel])

        greens = [sns.color_palette("Greens_d", 3)[1]]
        blues = 3 * [sns.color_palette("Blues_d", 3)[1]]
        flare = 3 * sns.color_palette("flare", 1)
        reds = 3 * [sns.color_palette("Reds_d", 3)[1]]
        greys = 3 * [sns.color_palette("Greys_d", 3)[1]]
        palette = greens + blues + flare + reds + greys

        size = (22, 10)
        fig = plt.figure(figsize=size)
        ax = sns.barplot(
            x=xlabel,
            y=ylabel,
            data=data,
            palette=palette,
            dodge=False,
            ci="sd",
            capsize=0.2,
            errcolor=error_color,
        )
        text_on_bars(
            ax,
            [
                *baseline_map_val,
                *rot_map_vals,
                *scale_map_vals,
                *crop_map_vals,
                *color_map_vals,
            ],
            lambda v: f"{v:.3f}%",
        )

        min_y, max_y = 92, 96
        plt.ylim((min_y, max_y))
        plt.tight_layout()

        legend = [
            Patch(color=greens[0], label="Baseline"),
            Patch(color=blues[0], label="Rotation"),
            Patch(color=flare[0], label="Scale"),
            Patch(color=reds[0], label="Safe Crop"),
            Patch(color=greys[0], label="Color Jitter"),
        ]
        plt.legend(handles=legend)

        plt.savefig(output / "yolo_online_aug_experiment.pdf")
        if show:
            plt.show()

    ####################################################################################
    ## YOLO GRID
    ####################################################################################
    lrs = [fsci(lr) for lr in (0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001)]
    ciou_32 = (94.124, 94.530, 94.842, 94.350, 93.700, 93.945, 94.011)
    ciou_64 = (95.679, 95.728, 95.885, 95.426, 95.640, 94.992, 95.512)
    eiou_0_32 = (94.258, 94.155, 94.745, 94.494, 94.554, 94.423, 94.672)
    eiou_0_64 = (96.217, 95.889, 95.935, 95.376, 95.355, 95.886, 95.991)
    eiou_05_32 = (93.835, 94.011, 94.312, 94.502, 94.418, 93.798, 94.019)
    eiou_05_64 = (95.424, 95.440, 95.422, 95.637, 95.289, 95.451, 95.435)

    def make_data(lrs, mAPs, bs, loss):
        data = [(loss, bs, lr, mAP) for lr, mAP in zip(lrs, mAPs)]
        return pd.DataFrame(
            data, columns=["Loss", "Batch Size", "Learning Rate", "mAP [%]"]
        )

    dciou_32 = make_data(lrs, ciou_32, 32, "CIoU")
    dciou_64 = make_data(lrs, ciou_64, 64, "CIoU")
    deiou_0_32 = make_data(lrs, eiou_0_32, 32, "EIoU@0")
    deiou_0_64 = make_data(lrs, eiou_0_64, 64, "EIoU@0")
    deiou_05_32 = make_data(lrs, eiou_05_32, 32, "EIoU@0.5")
    deiou_05_64 = make_data(lrs, eiou_05_64, 64, "EIoU@0.5")

    data = pd.concat(
        [dciou_32, dciou_64, deiou_0_32, deiou_0_64, deiou_05_32, deiou_05_64]
    )

    if yolo_grid_all:
        palette = [
            sns.color_palette("Blues_d", 3)[1],
            sns.color_palette("Greens", 3)[1],
        ]

        sns.set_style("whitegrid")
        size = (10, 6)
        fig = plt.figure(figsize=size)
        min_y, max_y = 93, 97
        sns.scatterplot(
            x="Learning Rate",
            y="mAP [%]",
            data=data,
            hue="Batch Size",
            palette=palette,
            sizes=len(data) * [50],
        )
        for idx, row in data.iterrows():
            x = row["Learning Rate"]
            y = row["mAP [%]"]
            bs = row["Batch Size"]

            loss = row["Loss"]
            if loss == "EIoU@0":
                t = "EIoU"
            else:
                t = loss

            if x == "1.0e-4" and bs == 32 and loss == "EIoU@0.5":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            elif x == "1.0e-3" and bs == 32 and loss == "EIoU@0.5":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            elif x == "2.5e-3" and bs == 64 and loss == "EIoU@0":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            elif x == "1.0e-3" and bs == 64 and loss == "EIoU@0":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            elif x == "1.0e-4" and bs == 64 and loss == "EIoU@0.5":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            elif x == "2.5e-4" and bs == 64 and loss == "EIoU@0.5":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            elif x == "5.0e-4" and bs == 64 and loss == "EIoU@0.5":
                plt.text(x, y, t, horizontalalignment="right", size="medium")
            else:
                plt.text(x, y, t, horizontalalignment="left", size="medium")

        plt.legend(loc="center left")
        plt.savefig(output / "yolo_grid_bs_compare.pdf")
        if show:
            plt.show()

    if yolo_grid_heat:
        results = np.array([ciou_64, eiou_0_64, eiou_05_64])
        yticks = ["CIoU", "EIoU", f"Focal-EIoU\n({gamma}=0.5)"]

        fig = plt.figure(figsize=(7, 3.5))
        ax = sns.heatmap(results, annot=True, fmt=".3f", square=True)
        ax.set_ylabel("Loss Function")
        ax.set_xlabel("Learning Rate")
        ax.set_xticklabels(lrs)
        ax.set_yticklabels(yticks)
        ax.tick_params(axis="y", labelrotation=25)

        plt.tight_layout()
        plt.savefig(output / "yolo_grid_heat.pdf")
        if show:
            plt.show()

    ###########################################
    ## TUNING INPUT SIZE
    ###########################################
    if yolo_input_size:
        sns.set(font_scale=2)
        sns.set_style("whitegrid")

        input_size = (544, 576, 608, 640, 672, 704, 736, 768, 800, 832)
        # test = ( 86.674, 87.695, 88.884, 91.531, 91.871, 92.227, 92.925, 93.506, 92.890, 92.885,)
        # fmt: off
        valid = (92.534, 95.237, 96.370, 96.210, 95.659, 96.541, 97.006, 96.571, 96.359, 95.791)
        # fmt: on

        valid = valid[2:]
        input_size = input_size[2:]
        input_size = [f"{isize}x{isize}" for isize in input_size]

        ylabel, xlabel = "mAP [%]", "Input Size [px x px]"
        data = pd.DataFrame({xlabel: input_size, ylabel: valid})

        min_y, max_y = 95, 97.5

        size = (17, 7)
        fig = plt.figure(figsize=size)
        ax = sns.barplot(
            x=xlabel, y=ylabel, data=data, palette=len(input_size) * [blue]
        )
        text_on_bars(ax, valid)
        plt.ylim((min_y, max_y))

        plt.savefig(output / "yolo_input_size_tuning.pdf")
        if show:
            plt.show()

    score_threshs = [f"{val:.2f}" for val in np.arange(0.1, 0.55, 0.05)]
    iou_threshs = [f"{val:.2f}" for val in np.arange(0.1, 0.55, 0.05)]
    if yolo_diou_heat:
        # fmt: off
        # x = score thresh
        # y = iou_thresh
        results = np.array([
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9700638651847839,0.9681911468505859,0.966210663318634,0.9659355282783508,0.9643896222114563,0.9620227813720703],
            [0.9703562259674072,0.9703562259674072,0.969967782497406,0.969967782497406,0.9680961966514587,0.9661168456077576,0.9658427834510803,0.9642968773841858,0.9619309306144714],
            [0.9703463912010193,0.9703463912010193,0.9699578881263733,0.9699578881263733,0.9680914282798767,0.9661144614219666,0.9658427834510803,0.9642968773841858,0.9619309306144714]
        ]) * 100
        # fmt: on
        make_score_tuning_heatmap(results, score_threshs, iou_threshs, size=(11, 11))

        plt.savefig(output / "yolo_diou_heat.pdf")
        if show:
            plt.show()

    if yolo_wbf_heat:
        # fmt: off
        # x = score thresh
        # y = iou_thresh
        results = np.array([
            [0.9697034358978271,0.9718723893165588,0.9712323546409607,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9697034358978271,0.9718723893165588,0.9712323546409607,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9697034358978271,0.9718723893165588,0.9712323546409607,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9697034358978271,0.9718723893165588,0.9712323546409607,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9697034358978271,0.9708917737007141,0.9712323546409607,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9697089791297913,0.9709010720252991,0.9712336659431458,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9697089791297913,0.9709010720252991,0.9712336659431458,0.9717895984649658,0.9707955718040466,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.969697892665863,0.9709005355834961,0.97123783826828,0.9717950224876404,0.9707998633384705,0.9694214463233948,0.9690998196601868,0.9657287001609802,0.9639574885368347],
            [0.9702332019805908,0.9712006449699402,0.9712073802947998,0.9715803265571594,0.9706361293792725,0.9691805839538574,0.9688961505889893,0.9652941823005676,0.96356600522995],
        ]) * 100
        # fmt: on
        make_score_tuning_heatmap(results, score_threshs, iou_threshs, size=(11, 11))

        plt.savefig(output / "yolo_wbf_heat.pdf")
        if show:
            plt.show()

    if yolo_diou_tta_heat:
        # fmt: off
        results = np.array([
            [0.9697441458702087,0.9697441458702087,0.9697441458702087,0.9697441458702087,0.9697441458702087,0.9697441458702087,0.9697441458702087,0.9697441458702087,0.9697441458702087],
            [0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341],
            [0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341],
            [0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341,0.9697427153587341],
            [0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432],
            [0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432],
            [0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432,0.9697363376617432],
            [0.9700061678886414,0.9700061678886414,0.9700061678886414,0.9700061678886414,0.9700061678886414,0.9696640372276306,0.9696640372276306,0.9696640372276306,0.9696640372276306],
            [0.9700602889060974,0.9699594378471375,0.9698143601417542,0.9698143601417542,0.9698143601417542,0.9694721698760986,0.9694721698760986,0.9694721698760986,0.9694721698760986],
        ])
        # fmt: on
        make_score_tuning_heatmap(results, score_threshs, iou_threshs, size=(11, 11))

        plt.savefig(output / "yolo_diou_tta_heat.pdf")
        if show:
            plt.show()

    if yolo_wbf_tta_heat:
        # fmt: off
        # x = score thresh
        # y = iou_thresh
        results = np.array([
            [0.9838979244232178,0.9833899140357971,0.9836897253990173,0.9842371940612793,0.9841075539588928,0.9838888049125671,0.9836196899414062,0.9824966788291931,0.9819128513336182],
            [0.9838979244232178,0.9833899140357971,0.9836897253990173,0.9842371940612793,0.9841075539588928,0.9838888049125671,0.9836196899414062,0.9824966788291931,0.9819128513336182],
            [0.9838977456092834,0.9833879470825195,0.9836878776550293,0.9842371940612793,0.9841063618659973,0.9838888049125671,0.9836196899414062,0.9824966788291931,0.9819128513336182],
            [0.9847074151039124,0.9833979606628418,0.9836926460266113,0.983997642993927,0.9841063618659973,0.9838888049125671,0.9836196899414062,0.9824966788291931,0.9819128513336182],
            [0.9846575856208801,0.9833466410636902,0.9836950302124023,0.9840002655982971,0.9841086268424988,0.9838888049125671,0.9836196899414062,0.9824972748756409,0.9819130897521973],
            [0.9846575856208801,0.9833466410636902,0.9836950302124023,0.9840002655982971,0.9841086268424988,0.9838888049125671,0.9836196899414062,0.9824972748756409,0.9819130897521973],
            [0.9846559166908264,0.9833500385284424,0.9836950302124023,0.9840012192726135,0.9841094613075256,0.9838873744010925,0.9836196899414062,0.9824957847595215,0.9819116592407227],
            [0.9854231476783752,0.9836974143981934,0.984059751033783,0.9840021133422852,0.9841108322143555,0.9838890433311462,0.9836180806159973,0.9824957847595215,0.9819116592407227],
            [0.9852632880210876,0.9834897518157959,0.9843370318412781,0.9838964343070984,0.9839127063751221,0.9830084443092346,0.9826939702033997,0.9814903736114502,0.9808415770530701],
        ]) * 100
        # fmt: on
        make_score_tuning_heatmap(results, score_threshs, iou_threshs, size=(11, 11))

        plt.savefig(output / "yolo_wbf_tta_heat.pdf")
        if show:
            plt.show()

    if yolo_wbf_tta_votes:
        sns.set_style("whitegrid")
        # using iou_thresh = 0.45, score_thresh = 0.1
        vote_thresh = [1, 2, 3, 4, 5, 6, 7, 8]
        # fmt: off
        results = np.array([0.98542315, 0.98542315, 0.9854284, 0.9854284, 0.98466796, 0.98466796, 0.98439574, 0.98188525]) * 100
        # fmt: on

        xlabel, ylabel = "Minimum Votes", "mAP [%]"

        data = pd.DataFrame({xlabel: vote_thresh, ylabel: results})

        min_y, max_y = 98, 98.8

        size = (8, 3)
        fig = plt.figure(figsize=size)
        ax = sns.barplot(
            x=xlabel, y=ylabel, data=data, palette=len(vote_thresh) * [blue]
        )
        text_on_bars(ax, results)
        plt.ylim((min_y, max_y))

        plt.tight_layout()
        plt.savefig(output / "yolo_wbf_tta_votes.pdf")
        if show:
            plt.show()

    if yolo_all_tuning:
        sns.set_style("whitegrid")
        # fmt: off
        types = ["DIoU Untuned, 608x608", "Input Size Tuned", "DIoU Tuned", "WBF Tuned", "WBF-TTA Tuned"]
        valid = [v*100 for v in [0.9637025, 0.97006387, 0.9703562, 0.9718724, 0.9854284]]
        test = [v*100 for v in [0.88883495, 0.92925644, 0.92925644, 0.9318778, 0.95491886]]
        experiments = ["Baseline", "InputSize", "DIoU-Tune", "WBF-Tune", "WBF-TTA-Tune"]
        # fmt: on

        ylabel, xlabel = "mAP [%]", "Performed Tuning Experiment\n\n"

        n_exps = len(valid)
        dataset_type = n_exps * ["Validation"] + n_exps * ["Test"]
        experiments = 2 * experiments
        maps = valid + test

        data = pd.DataFrame(
            {xlabel: experiments, "Dataset": dataset_type, ylabel: maps}
        )
        palette = [blue, green]

        ax = sns.catplot(
            x=xlabel,
            y=ylabel,
            hue="Dataset",
            kind="bar",
            palette=palette,
            data=data,
            aspect=1.7,
            legend=False,
        )
        min_y, max_y = 88, 100
        plt.ylim((min_y, max_y))
        plt.legend(loc="upper left")

        text_on_bars(ax.ax, valid + test, formatter=lambda v: f"{v:.3f}%")

        plt.tight_layout()
        plt.savefig(output / "yolo_all_tuning.pdf")
        if show:
            plt.show()

    ####################################################################################
    ## MUNET EXPERIMENTS
    ####################################################################################

    ###########################################
    ## LEARNING RATE SEARCH
    ###########################################
    if munet_lr:
        sns.set_style("whitegrid")

        results = (
            (0.01, (83.29903, 82.89061, 83.01646)),
            (0.005, (82.49348, 82.85790, 81.51328)),
            (0.0025, (82.72171, 81.99200, 81.93025)),
            (0.001, (83.12994, 82.57712, 83.19931)),
            (0.0005, (82.69577, 82.94604, 82.57509)),
            (0.00025, (82.55172, 82.35741, 82.19239)),
            (0.0001, (81.68054, 81.44921, 81.67149)),
        )

        ylabel, xlabel = "f1-Score [%]", "Learning Rate"

        data = []
        for lr, f1s in results:
            for f1 in f1s:
                data.append((fsci(lr), f1))

        data = pd.DataFrame(data, columns=[xlabel, ylabel])

        # size = (8, 5)
        size = (22, 10)
        fig = plt.figure(figsize=size)
        ax = sns.barplot(
            x=xlabel,
            y=ylabel,
            data=data,
            palette=len(results) * [blue],
            capsize=0.2,
            ci="sd",
            errcolor=error_color,
        )
        ax.set_ylabel(ylabel, fontsize=34)
        ax.set_xlabel(xlabel, fontsize=34)
        ax.tick_params(labelsize=28)
        text_on_bars(ax, [np.mean(r) for _, r in results], fontsize=28)
        sns.set_style("whitegrid")
        # set_capcolor(ax, "black")
        plt.ylim((80, 84))

        plt.savefig(output / "munet_lr_exp.pdf")
        if show:
            plt.show()

    ###########################################
    ## OFFLINE AUG
    ###########################################
    if munet_offline:
        sns.set_style("whitegrid")
        configs = ("Baseline", "R", "F", "F,R", "P", "P,R", "P,F", "P,F,R")
        maps = (73.415, 87.140, 84.415, 91.113, 79.403, 89.641, 86.129, 92.578)

        results = (
            ("Baseline", (83.29903, 82.89061, 83.01646)),
            ("R", (83.45442, 83.63076, 83.82413)),
            ("F", (83.89591, 84.56100, 83.51876)),
            ("F,R", (83.46847, 84.46586, 83.15600)),
            ("P", (85.23983, 84.51490, 84.26195)),
            ("P,R", (85.09336, 84.42223, 84.59901)),
            ("P,F", (84.11455, 84.38198, 84.43308)),
            ("P,F,R", (85.50255, 84.88704, 84.33488)),
        )

        by_mean = lambda v: np.array(v[1]).mean()
        results = sorted(results, key=by_mean)

        ylabel, xlabel = "f1-Score [%]", "Configuration"

        data = []
        for config, f1s in results:
            for f1 in f1s:
                data.append((config, f1))
        data = pd.DataFrame(data, columns=[xlabel, ylabel])

        palette = sns.color_palette([green] + (len(results) - 1) * [blue], len(results))

        # size = (8, 5)
        size = (22, 10)
        fig = plt.figure(figsize=size)
        ax = sns.barplot(
            x=xlabel,
            y=ylabel,
            data=data,
            palette=palette,
            capsize=0.2,
            ci="sd",
            errcolor=error_color,
        )
        # set_capcolor(ax, "black")

        ax.set_ylabel(ylabel, fontsize=34)
        ax.set_xlabel(xlabel, fontsize=34)
        ax.tick_params(labelsize=28)
        text_on_bars(ax, [np.mean(r) for _, r in results], fontsize=28)

        min_y, max_y = 82, 86
        plt.ylim((min_y, max_y))

        # ticks
        # yticks = np.arange(min_y, max_y, step)
        # ax.set_yticks(yticks)

        plt.rcParams["legend.fontsize"] = 28
        legend = [
            Patch(color="white", label="P: Projection"),
            Patch(color="white", label="F: Horizontal Flip"),
            Patch(color="white", label="R: Rotation"),
            Patch(color="white", label="Baseline: LR = 1e-2"),
        ]
        plt.legend(handles=legend)

        plt.savefig(output / "munet_offline_aug_experiment.pdf")
        if show:
            plt.show()

    ###########################################
    ## ONLINE AUG
    ###########################################
    if munet_online:
        augs = ("Rotation", "Scale", "Crop", "ColorJitter")

        baseline = (("Baseline", (85.50255, 84.88704, 84.33488)),)

        rot_results = (
            ("10°", (84.23689, 85.06489, 84.85812)),
            ("20°", (84.96628, 85.36835, 84.53506)),
            ("30°", (83.77186, 84.65420, 83.69778)),
        )

        scale_results = (
            ("10%", (83.55176, 83.19931, 83.40881)),
            ("20%", (83.20576, 83.07797, 83.35737)),
            ("30%", (83.08891, 82.81269, 84.58560)),
        )

        crop_results = (
            ("70%", (84.47065, 84.75834, 84.81954)),
            ("80%", (85.18958, 84.89105, 84.99763)),
            ("90%", (85.73553, 85.57474, 85.80949)),
        )

        color_results = (
            ("10%\uFEFF", (85.39170, 85.19188, 86.07763)),
            ("20%\uFEFF", (85.37144, 85.89094, 85.76421)),
            ("30%\uFEFF", (85.31983, 84.56982, 85.53279)),
        )

        ylabel, xlabel = "f1-Score [%]", "Parameters"
        sns.set(font_scale=2)
        sns.set_style("whitegrid")

        data = []

        def to_data(raw, type_):
            for p1, p2 in raw:
                for p2_ in p2:
                    data.append((type_, p1, p2_))

        to_data(baseline, "Baseline")
        to_data(rot_results, "Rotation")
        to_data(scale_results, "Scale")
        to_data(crop_results, "Crop")
        to_data(color_results, "ColorJitter")
        data = pd.DataFrame(data, columns=["Type", xlabel, ylabel])

        flare = 3 * sns.color_palette("flare", 1)
        blues = 3 * [sns.color_palette("Blues_d", 3)[1]]
        greens = [sns.color_palette("Greens_d", 3)[1]]
        reds = 3 * [sns.color_palette("Reds_d", 3)[1]]
        greys = 3 * [sns.color_palette("Greys_d", 3)[1]]
        palette = greens + blues + flare + reds + greys

        size = (22, 10)
        fig = plt.figure(figsize=size)
        ax = sns.barplot(
            x=xlabel,
            y=ylabel,
            data=data,
            palette=palette,
            dodge=False,
            capsize=0.2,
            ci="sd",
            errcolor=error_color,
        )
        # set_capcolor(ax, "black")

        def m(res):
            return [np.mean(r) for _, r in res]

        text_on_bars(
            ax,
            m(baseline)
            + m(rot_results)
            + m(scale_results)
            + m(crop_results)
            + m(color_results),
        )

        min_y, max_y = 80, 88
        plt.ylim((min_y, max_y))
        plt.tight_layout()

        legend = [
            Patch(color=greens[0], label="Baseline"),
            Patch(color=blues[0], label="Rotation"),
            Patch(color=flare[0], label="Scale"),
            Patch(color=reds[0], label="Crop"),
            Patch(color=greys[0], label="Color Jitter"),
        ]
        plt.legend(handles=legend)

        plt.savefig(output / "munet_online_aug_experiment.pdf")
        if show:
            plt.show()

    # fmt: off
    res = (
        [32, "focal2_0.1", 0.01    , 84.3573,  81.4203,  85.6070,  85.3463,  80.0393,  87.6773,  83.3993,  82.9663,  83.6583],
        [32, "focal2_0.1", 0.005   , 84.2307,  82.5577,  84.9237,  85.9130,  80.9737,  88.0820,  82.6600,  84.2353,  82.0827],
        [32, "focal2_0.1", 0.0025  , 83.2997,  81.3367,  84.1007,  80.1717,  74.4970,  82.6633,  86.6923,  89.6437,  85.5930],
        [32, "focal2_0.1", 0.001   , 83.6880,  82.0747,  84.3487,  81.4933,  76.1917,  83.8217,  86.0047,  88.9737,  84.8873],
        [32, "focal2_0.1", 0.0005  , 84.1557,  83.2650,  84.5193,  85.1330,  80.0820,  87.3513,  83.2187,  86.7240,  81.8870],
        [32, "focal2_0.1", 0.00025 , 84.0703,  83.6063,  84.2553,  85.0503,  79.9430,  87.2937,  83.1207,  87.6593,  81.4247],
        [32, "focal2_0.1", 0.0001  , 83.1267,  83.7790,  82.8600,  86.2730,  82.6813,  87.8503,  80.2027,  84.9120,  78.4067],
        [32, "focal2_0.8", 0.01    , 84.5143,  82.6660,  85.2760,  85.3493,  79.6097,  87.8700,  83.7067,  86.0030,  82.8453],
        [32, "focal2_0.8", 0.005   , 84.6240,  83.0110,  85.2887,  84.0923,  78.8143,  86.4103,  85.1727,  87.6950,  84.2047],
        [32, "focal2_0.8", 0.0025  , 83.4513,  82.3743,  83.8903,  81.1487,  75.9403,  83.4360,  85.9127,  90.0563,  84.3650],
        [32, "focal2_0.8", 0.001   , 83.3273,  81.7247,  83.9787,  81.7593,  75.9390,  84.3153,  85.0273,  88.5890,  83.7023],
        [32, "focal2_0.8", 0.0005  , 83.5700,  83.0133,  83.7967,  83.2260,  78.4140,  85.3393,  83.9257,  88.1947,  82.3233],
        [32, "focal2_0.8", 0.00025 , 83.7633,  83.8423,  83.7307,  84.2000,  79.8040,  86.1307,  83.3327,  88.3113,  81.4630],
        [32, "focal2_0.8", 0.0001  , 82.8853,  83.7757,  82.5220,  85.8127,  82.2730,  87.3670,  80.1537,  85.3467,  78.1873],
        [32, "dice",       0.01    , 85.4490,  84.4453,  85.8817,  88.2673,  84.6293,  89.8650,  82.8120,  84.3217,  82.2777],
        [32, "dice",       0.005   , 84.9513,  82.8607,  85.8253,  86.7547,  81.8163,  88.9237,  83.2367,  84.0073,  82.9443],
        [32, "dice",       0.0025  , 85.0003,  83.5350,  85.6063,  87.4727,  82.4123,  89.6950,  82.6720,  84.7373,  81.8763],
        [32, "dice",       0.001   , 83.7517,  84.9903,  83.2563,  87.3810,  83.0063,  89.3020,  80.4143,  87.0733,  77.9797],
        [32, "dice",       0.0005  , 79.6540,  81.1443,  79.0317,  94.2430,  92.6613,  94.9373,  68.9777,  72.1760,  67.6917],
        [32, "dice",       0.00025 , 71.0880,  72.1793,  70.6240,  97.8740,  97.3113,  98.1210,  55.8143,  57.3657,  55.1670],
        [32, "dice",       0.0001  , 61.2197,  60.1760,  61.6860,  99.1407,  98.5847,  99.3847,  44.2830,  43.3090,  44.7230],
        [64, "focal2_0.1", 0.01    , 84.5230,  82.6357,  85.3103,  85.3540,  80.4230,  87.5193,  83.7830,  85.0000,  83.3103],
        [64, "focal2_0.1", 0.005   , 84.3567,  82.6490,  85.0347,  83.2203,  76.5050,  86.1697,  85.5590,  89.9877,  83.9537],
        [64, "focal2_0.1", 0.0025  , 83.8303,  81.9410,  84.6063,  82.1687,  76.5887,  84.6190,  85.5990,  88.1197,  84.6403],
        [64, "focal2_0.1", 0.001   , 83.3823,  81.7557,  84.0480,  82.2147,  76.3297,  84.7983,  84.7400,  88.3033,  83.4380],
        [64, "focal2_0.1", 0.0005  , 84.0877,  83.2887,  84.4093,  84.1043,  78.4090,  86.6057,  84.0867,  88.8840,  82.3290],
        [64, "focal2_0.1", 0.00025 , 84.0090,  83.5783,  84.1867,  84.4883,  79.4037,  86.7210,  83.5360,  88.2347,  81.8003],
        [64, "focal2_0.1", 0.0001  , 83.4527,  84.1140,  83.1827,  86.3100,  82.8000,  87.8510,  80.7807,  85.4837,  78.9860],
        [64, "focal2_0.8", 0.01    , 84.8790,  83.9550,  85.2640,  84.9900,  80.2740,  87.0613,  84.8933,  88.0533,  83.6987],
        [64, "focal2_0.8", 0.005   , 84.2067,  83.4243,  84.5330,  83.6010,  79.8820,  85.2343,  84.8923,  87.4043,  83.9020],
        [64, "focal2_0.8", 0.0025  , 83.0467,  81.9543,  83.4857,  81.4737,  76.0850,  83.8397,  84.8847,  89.2283,  83.2860],
        [64, "focal2_0.8", 0.001   , 83.8487,  82.7177,  84.3003,  83.0420,  76.8843,  85.7467,  84.7430,  89.6550,  82.9577],
        [64, "focal2_0.8", 0.0005  , 84.1937,  84.1330,  84.2150,  84.0663,  79.3457,  86.1393,  84.3260,  89.5440,  82.3860],
        [64, "focal2_0.8", 0.00025 , 83.8553,  84.0170,  83.7900,  84.9910,  80.8113,  86.8267,  82.7550,  87.4973,  80.9620],
        [64, "focal2_0.8", 0.0001  , 83.1697,  83.9473,  82.8530,  85.5173,  81.9420,  87.0870,  80.9500,  86.0580,  79.0127],
        [64, "dice",       0.01    , 84.1380,  82.6873,  84.7630,  86.6617,  82.1410,  88.6467,  81.8227,  83.2630,  81.3333],
        [64, "dice",       0.005   , 84.5200,  83.3547,  85.0127,  86.6337,  82.1583,  88.5987,  82.5250,  84.5867,  81.7410],
        [64, "dice",       0.0025  , 84.3860,  83.4197,  84.7950,  84.3993,  79.9007,  86.3747,  84.5423,  87.3567,  83.4843],
        [64, "dice",       0.001   , 83.9007,  85.2167,  83.3727,  86.8687,  82.7877,  88.6603,  81.1393,  87.8150,  78.6880],
        [64, "dice",       0.0005  , 79.5757,  81.0253,  78.9717,  94.0173,  92.2937,  94.7740,  68.9807,  72.2097,  67.6863],
        [64, "dice",       0.00025 , 70.6473,  71.5100,  70.2790,  98.1930,  97.6240,  98.4433,  55.1707,  56.4200,  54.6453],
        [64, "dice",       0.0001  , 61.1427,  59.6127,  61.8363,  99.2790,  98.9110,  99.4413,  44.1743,  42.6633,  44.8690],
    )
    munet_grid_data = pd.DataFrame(res, columns=["Batch Size", "Loss", "Learning Rate", "F1 B", "F1 C", "F1 U", "R B", "R C", "R U", "P B", "P C", "P U"])

    # fmt: on

    lrs = [fsci(lr) for lr in [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]]
    if munet_grid_all_bs:
        ylabel, xlabel = "f1-Score [%]", "Learning Rate"

        palette = [
            sns.color_palette("Blues_d", 3)[1],
            sns.color_palette("Greens", 3)[1],
        ]

        data[xlabel] = 6 * lrs
        print(data)

        size = (7, 4)
        fig = plt.figure(figsize=size)
        # min_y, max_y = 93, 97
        sns.scatterplot(
            x=xlabel, y=ylabel, data=data, hue="Batch Size", palette=palette
        )

        plt.savefig(output / "munet_grid_bs_compare.pdf")
        if show:
            plt.show()

    def loss_to_tick(loss):
        if loss == "dice":
            return "Dice"

        if loss == "focal2_0.1":
            return f"Focal\n{alpha}=0.1\n{gamma}=2"

        if loss == "focal2_0.8":
            return f"Focal\n{alpha}=0.8\n{gamma}=2"

    if munet_grid_heat:
        ylabel, xlabel = "f1-Score [%]", "Learning Rate"
        sns.set_style("whitegrid")

        def loss_to_tick(loss):
            if loss == "dice":
                return "Dice"

            if loss == "focal2_0.1":
                return f"Focal\n{alpha}=0.1\n{gamma}=2"

            if loss == "focal2_0.8":
                return f"Focal\n{alpha}=0.8\n{gamma}=2"

        def to_ytick(loss, bs):
            return f"{loss_to_tick(loss)}\nBS {bs}"

        def munet_heat(batch_sizes, losses, ylabel, metric):
            data = []
            yticks = []
            for bs in batch_sizes:
                for loss in losses:
                    conf = munet_grid_data.loc[munet_grid_data["Batch Size"] == bs]
                    conf = conf.loc[conf["Loss"] == loss]
                    conf["Learning Rate"] = conf["Learning Rate"].apply(fsci)
                    m = conf[metric]
                    data.append(m.values)
                    yticks.append(to_ytick(loss, bs))
            data = np.array(data)

            size = (11, 11)
            fig = plt.figure(figsize=size)
            ax = sns.heatmap(data, annot=True, fmt=".3f", square=True)

            ax.set_xlabel(xlabel)
            ax.set_xlabel(ylabel)
            ax.set_xticklabels(lrs)
            ax.set_yticklabels(yticks)
            ax.tick_params(axis="y", labelrotation=25)

            plt.tight_layout()
            plt.savefig(output / f"munet_grid_heat_{metric.replace(' ', '_')}.pdf")
            if show:
                plt.show()

        batch_sizes = [32, 64]
        losses = ["focal2_0.1", "focal2_0.8", "dice"]

        metrics_ylabel = (
            ("F1 B", "f1-Score [%]"),
            ("F1 C", "f1-Score [%]"),
            ("R B", "Recall [%]"),
            ("R C", "Recall [%]"),
            ("P B", "Precision [%]"),
            ("P C", "Precision [%]"),
        )

        for metric, ylabel in metrics_ylabel:
            munet_heat(batch_sizes, losses, ylabel, metric)

    if munet_folds:
        # fmt: off
        res = (
            ((32, "focal2_0.1", 0.01   , "F B"),  ((83.872,81.375,84.885), (85.468,78.494,88.531), (82.334,84.477,81.529)), ((74.872,69.550,81.328), (90.822,89.238,92.526), (63.687,56.979,72.548))),
            ((32, "focal2_0.1", 0.0001 , "R B" ), ((83.427,84.136,83.137), (86.754,83.093,88.362), (80.345,85.207,78.495)), ((79.264,77.716,80.924), (90.923,89.007,92.984), (70.255,68.967,71.634))),
            ((32, "focal2_0.1", 0.0025 , "P B" ), ((82.975,81.717,83.489), (79.242,74.103,81.499), (87.078,91.075,85.578)), ((76.815,72.304,82.249), (85.250,84.603,85.945), (69.899,63.126,78.858))),
            ((32, "focal2_0.1", 0.0001 , "F C"),  ((83.427,84.136,83.137), (86.754,83.093,88.362), (80.345,85.207,78.495)), ((79.264,77.716,80.924), (90.923,89.007,92.984), (70.255,68.967,71.634))),
            ((32, "focal2_0.1", 0.0001 , "R C" ), ((83.427,84.136,83.137), (86.754,83.093,88.362), (80.345,85.207,78.495)), ((79.264,77.716,80.924), (90.923,89.007,92.984), (70.255,68.967,71.634))),
            ((32, "focal2_0.1", 0.0025 , "P C" ), ((82.975,81.717,83.489), (79.242,74.103,81.499), (87.078,91.075,85.578)), ((76.815,72.304,82.249), (85.250,84.603,85.945), (69.899,63.126,78.858))),
            ((32, "focal2_0.8", 0.005  , "F B"),  ((85.356,84.684,85.633), (83.931,79.539,85.860), (86.830,90.540,85.406)), ((77.864,73.536,83.038), (88.799,88.118,89.533), (69.326,63.095,77.422))),
            ((32, "focal2_0.8", 0.0001 , "R B" ), ((82.897,84.193,82.364), (86.202,83.593,87.347), (79.836,84.802,77.919)), ((79.154,77.904,80.497), (90.094,88.630,91.668), (70.583,69.494,71.753))),
            ((32, "focal2_0.8", 0.0025 , "P B" ), ((82.858,81.779,83.293), (79.582,73.883,82.085), (86.416,91.564,84.537)), ((77.590,73.677,82.040), (84.508,82.389,86.787), (71.719,66.631,77.785))),
            ((32, "focal2_0.8", 0.00025, "F C"),  ((83.831,84.469,83.570), (84.067,80.533,85.619), (83.595,88.809,81.616)), ((81.271,80.063,82.539), (88.658,86.310,91.184), (75.020,74.660,75.391))),
            ((32, "focal2_0.8", 0.0001 , "R C" ), ((82.897,84.193,82.364), (86.202,83.593,87.347), (79.836,84.802,77.919)), ((79.154,77.904,80.497), (90.094,88.630,91.668), (70.583,69.494,71.753))),
            ((32, "focal2_0.8", 0.0025 , "P C" ), ((82.858,81.779,83.293), (79.582,73.883,82.085), (86.416,91.564,84.537)), ((77.590,73.677,82.040), (84.508,82.389,86.787), (71.719,66.631,77.785))),
            ((32, "dice"      , 0.01   , "F B"),  ((85.642,83.823,86.442), (87.954,86.140,88.751), (83.448,81.627,84.250)), ((75.260,69.917,81.997), (93.127,93.115,93.139), (63.145,55.972,73.236))),
            ((32, "dice"      , 0.0001 , "R B" ), ((60.812,59.636,61.340), (99.286,98.986,99.417), (43.828,42.673,44.353)), ((51.920,49.812,54.363), (99.091,98.489,99.739), (35.175,33.336,37.365))),
            ((32, "dice"      , 0.005  , "P B" ), ((85.299,83.276,86.127), (85.958,79.875,88.629), (84.650,86.979,83.763)), ((79.028,76.362,81.993), (90.237,88.598,92.000), (70.296,67.096,73.950))),
            ((32, "dice"      , 0.001  , "F C"),  ((83.843,85.137,83.326), (87.343,83.000,89.250), (80.613,87.387,78.140)), ((80.550,79.434,81.787), (92.682,92.707,92.654), (71.227,69.486,73.201))),
            ((32, "dice"      , 0.0001 , "R C" ), ((60.812,59.636,61.340), (99.286,98.986,99.417), (43.828,42.673,44.353)), ((51.920,49.812,54.363), (99.091,98.489,99.739), (35.175,33.336,37.365))),
            ((32, "dice"      , 0.001  , "P C" ), ((83.843,85.137,83.326), (87.343,83.000,89.250), (80.613,87.387,78.140)), ((80.550,79.434,81.787), (92.682,92.707,92.654), (71.227,69.486,73.201))),
            ((64, "focal2_0.1", 0.01   , "F B"),  ((84.698,83.185,85.334), (84.122,80.070,85.901), (85.283,86.551,84.775)), ((75.542,70.765,81.119), (87.930,85.622,90.413), (66.213,60.302,73.557))),
            ((64, "focal2_0.1", 0.0001 , "R B" ), ((83.128,83.523,82.964), (86.418,83.222,87.821), (80.079,83.826,78.616)), ((79.524,77.950,81.219), (91.210,89.478,93.072), (70.492,69.053,72.044))),
            ((64, "focal2_0.1", 0.0025 , "P B" ), ((83.756,81.333,84.760), (80.579,75.154,82.961), (87.194,88.618,86.640)), ((79.833,76.911,83.067), (85.412,83.431,87.543), (74.938,71.337,79.027))),
            ((64, "focal2_0.1", 0.0001 , "F C"),  ((83.627,84.759,83.168), (86.346,82.765,87.918), (81.075,86.851,78.905)), ((79.941,78.476,81.499), (90.566,88.421,92.873), (71.548,70.542,72.607))),
            ((64, "focal2_0.1", 0.0001 , "R C" ), ((83.128,83.523,82.964), (86.418,83.222,87.821), (80.079,83.826,78.616)), ((79.524,77.950,81.219), (91.210,89.478,93.072), (70.492,69.053,72.044))),
            ((64, "focal2_0.1", 0.005  , "P C" ), ((84.529,82.222,85.441), (81.751,73.864,85.215), (87.501,92.713,85.668)), ((79.602,76.872,82.644), (87.110,85.547,88.790), (73.286,69.795,77.293))),
            ((64, "focal2_0.8", 0.01   , "F B"),  ((85.284,85.184,85.325), (86.905,83.037,88.604), (83.722,87.445,82.280)), ((78.042,74.622,82.089), (91.779,91.793,91.764), (67.881,62.863,74.260))),
            ((64, "focal2_0.8", 0.0001 , "R B" ), ((83.256,84.452,82.770), (85.931,82.578,87.403), (80.742,86.412,78.602)), ((79.851,78.326,81.476), (89.848,87.699,92.160), (71.856,70.763,73.011))),
            ((64, "focal2_0.8", 0.01   , "P B" ), ((84.597,82.386,85.526), (81.511,76.930,83.523), (87.927,88.675,87.627)), ((78.358,74.724,82.551), (86.090,84.855,87.418), (71.901,66.753,78.197))),
            ((64, "focal2_0.8", 0.0005 , "F C"),  ((83.974,84.773,83.645), (83.540,80.679,84.796), (84.413,89.304,82.524)), ((81.080,79.644,82.591), (86.993,84.574,89.594), (75.919,75.257,76.603))),
            ((64, "focal2_0.8", 0.0001 , "R C" ), ((83.256,84.452,82.770), (85.931,82.578,87.403), (80.742,86.412,78.602)), ((79.851,78.326,81.476), (89.848,87.699,92.160), (71.856,70.763,73.011))),
            ((64, "focal2_0.8", 0.001  , "P C" ), ((83.804,81.819,84.597), (80.739,73.748,83.810), (87.110,91.875,85.398)), ((81.956,80.473,83.514), (85.692,83.193,88.379), (78.532,77.926,79.157))),
            ((64, "dice"      , 0.005  , "F B"),  ((84.684,85.230,84.460), (87.422,84.001,88.924), (82.113,86.495,80.423)), ((73.236,67.619,80.554), (91.831,92.573,91.032), (60.903,53.262,72.239))),
            ((64, "dice"      , 0.0001 , "R B" ), ((60.793,59.268,61.485), (99.374,99.155,99.471), (43.791,42.266,44.494)), ((52.078,50.137,54.311), (99.150,98.537,99.810), (35.313,33.623,37.305))),
            ((64, "dice"      , 0.0025 , "P B" ), ((85.096,83.452,85.782), (81.981,77.554,83.924), (88.458,90.321,87.724)), ((81.533,80.229,82.968), (86.996,86.525,87.503), (76.716,74.787,78.881))),
            ((64, "dice"      , 0.001  , "F C"),  ((83.883,85.242,83.334), (87.787,84.049,89.428), (80.311,86.469,78.018)), ((80.005,78.705,81.446), (92.721,92.543,92.912), (70.356,68.466,72.500))),
            ((64, "dice"      , 0.0001 , "R C" ), ((60.793,59.268,61.485), (99.374,99.155,99.471), (43.791,42.266,44.494)), ((52.078,50.137,54.311), (99.150,98.537,99.810), (35.313,33.623,37.305))),
            ((64, "dice"      , 0.001  , "P C" ), ((83.936,85.215,83.424), (86.023,81.905,87.831), (81.948,88.804,79.437)), ((80.728,79.378,82.230), (91.823,91.758,91.892), (72.025,69.941,74.406))),
        )
        data = []
        for (bs, loss, lr, metric), vvalues, tvalues in res:
            vvalues, tvalues = np.array(vvalues), np.array(tvalues)

            vf, vr, vp = vvalues
            tf, tr, tp = tvalues

            data.append((bs, loss, lr, metric, vf[0], tf[0], vr[0], tr[0], vp[0], tp[0]))

        data = pd.DataFrame(
            data,
            columns=["Batch Size", "Loss", "Learning Rate", "Metric", "FV", "FT", "RV", "RT", "PV", "PT"]
        )
        # fmt: on

        # sns.set(font_scale=2)
        sns.set_style("whitegrid")

        xlabel = "Learning Rate"
        ylabel = "f1-Score [%]"

        def loss_to_tick(loss):
            if loss == "dice":
                return "Dice"

            if loss == "focal2_0.1":
                return f"Focal: {alpha}=0.1, {gamma}=2"

            if loss == "focal2_0.8":
                return f"Focal: {alpha}=0.8, {gamma}=2"

        full_data = None
        for bs in (32, 64):
            full_data = None
            for loss in ("focal2_0.1", "focal2_0.8", "dice"):
                # print("Fold:", loss, bs)
                d = data
                d = d.loc[d["Loss"] == loss]
                d = d.loc[d["Batch Size"] == bs]
                d = d.drop_duplicates(subset=["Learning Rate"])
                # d = d.sort_values("Learning Rate", ascending=False)
                # d["Learning Rate"] = d["Learning Rate"].apply(fsci)

                nd = pd.DataFrame(
                    {
                        xlabel: 2 * d["Learning Rate"].to_list(),
                        ylabel: d["FV"].to_list() + d["FT"].to_list(),
                        # "Hue": len(d["FV"]) * [f"Valid{bs}{loss}"] + len(d["FT"]) * [f"Test{bs}{loss}"],
                        "Fold": 2 * len(d["FV"]) * [f"{loss_to_tick(loss)}"],
                        "Dataset": len(d["FV"]) * ["Validation"]
                        + len(d["FV"]) * ["Test"],
                    }
                )

                if full_data is None:
                    full_data = nd
                else:
                    full_data = pd.concat([full_data, nd], axis=0)

            # print(full_data)
            full_data = full_data.sort_values(
                ["Learning Rate", "Fold"], ascending=False
            )
            full_data["Learning Rate"] = full_data["Learning Rate"].apply(fsci)

            valid_data = full_data.loc[full_data["Dataset"] == "Validation"]
            test_data = full_data.loc[full_data["Dataset"] == "Test"]

            # fold_order = full_data["Fold"].unique()
            # fold_order.sort()
            # fold_order = [[o] for o in fold_order]
            print(full_data)
            full_data = full_data.sort_values(by="Fold")

            palette = [blue, green]
            ax = sns.catplot(
                # xlabel=6 * [xlabel],
                # ylabel=6 * [ylabel],
                x=xlabel,
                y=ylabel,
                kind="bar",
                hue="Dataset",
                row="Fold",
                data=full_data,
                order=lrs,
                hue_order=["Validation", "Test"],
                aspect=2 / 1,
                palette=palette,
                legend=True,
                legend_out=False,
            )
            min_y, max_y = 50, 93
            plt.ylim((min_y, max_y))

            for idx, (fold, ax_) in enumerate(
                zip(full_data["Fold"].unique(), ax.axes)
            ):
                vfold = valid_data.loc[valid_data["Fold"] == fold].copy()
                tfold = test_data.loc[test_data["Fold"] == fold].copy()

                vf1 = []
                for lr in lrs:
                    current = vfold.loc[vfold["Learning Rate"] == lr]
                    if current.empty:
                        vf1.append(0)
                    else:
                        vf1.append(current[ylabel].values[0])

                tf1 = []
                for lr in lrs:
                    current = tfold.loc[tfold["Learning Rate"] == lr]
                    if current.empty:
                        tf1.append(0)
                    else:
                        tf1.append(current[ylabel].values[0])

                text_on_bars(ax_[0], vf1 + tf1)

            # ax.legend(loc="upper right")
            # ax.axes[0].legend(loc="upper right")

            # text_on_bars(, valid_f1 + test_f1, formatter=lambda v: f"{v:.3f}%")
            # print(ax.axes)

            plt.tight_layout()
            plt.savefig(output / f"munet_folds_bs{bs}.pdf")
            if show:
                plt.show()


if __name__ == "__main__":
    main()


########################################################################################
## ALL DATA
########################################################################################


################################################
## DIoU
################################################
# iou_thresh,score_thresh,mAP@0.5:0.75
# 0.1,0.1,0.9700638651847839
# 0.1,0.15,0.9700638651847839
# 0.1,0.2,0.9700638651847839
# 0.1,0.25,0.9700638651847839
# 0.1,0.3,0.9681911468505859
# 0.1,0.35,0.966210663318634
# 0.1,0.4,0.9659355282783508
# 0.1,0.45,0.9643896222114563
# 0.1,0.5,0.9620227813720703
# 0.15,0.1,0.9700638651847839
# 0.15,0.15,0.9700638651847839
# 0.15,0.2,0.9700638651847839
# 0.15,0.25,0.9700638651847839
# 0.15,0.3,0.9681911468505859
# 0.15,0.35,0.966210663318634
# 0.15,0.4,0.9659355282783508
# 0.15,0.45,0.9643896222114563
# 0.15,0.5,0.9620227813720703
# 0.2,0.1,0.9700638651847839
# 0.2,0.15,0.9700638651847839
# 0.2,0.2,0.9700638651847839
# 0.2,0.25,0.9700638651847839
# 0.2,0.3,0.9681911468505859
# 0.2,0.35,0.966210663318634
# 0.2,0.4,0.9659355282783508
# 0.2,0.45,0.9643896222114563
# 0.2,0.5,0.9620227813720703
# 0.25,0.1,0.9700638651847839
# 0.25,0.15,0.9700638651847839
# 0.25,0.2,0.9700638651847839
# 0.25,0.25,0.9700638651847839
# 0.25,0.3,0.9681911468505859
# 0.25,0.35,0.966210663318634
# 0.25,0.4,0.9659355282783508
# 0.25,0.45,0.9643896222114563
# 0.25,0.5,0.9620227813720703
# 0.3,0.1,0.9700638651847839
# 0.3,0.15,0.9700638651847839
# 0.3,0.2,0.9700638651847839
# 0.3,0.25,0.9700638651847839
# 0.3,0.3,0.9681911468505859
# 0.3,0.35,0.966210663318634
# 0.3,0.4,0.9659355282783508
# 0.3,0.45,0.9643896222114563
# 0.3,0.5,0.9620227813720703
# 0.35,0.1,0.9700638651847839
# 0.35,0.15,0.9700638651847839
# 0.35,0.2,0.9700638651847839
# 0.35,0.25,0.9700638651847839
# 0.35,0.3,0.9681911468505859
# 0.35,0.35,0.966210663318634
# 0.35,0.4,0.9659355282783508
# 0.35,0.45,0.9643896222114563
# 0.35,0.5,0.9620227813720703
# 0.4,0.1,0.9700638651847839
# 0.4,0.15,0.9700638651847839
# 0.4,0.2,0.9700638651847839
# 0.4,0.25,0.9700638651847839
# 0.4,0.3,0.9681911468505859
# 0.4,0.35,0.966210663318634
# 0.4,0.4,0.9659355282783508
# 0.4,0.45,0.9643896222114563
# 0.4,0.5,0.9620227813720703
# 0.45,0.1,0.9703562259674072
# 0.45,0.15,0.9703562259674072
# 0.45,0.2,0.969967782497406
# 0.45,0.25,0.969967782497406
# 0.45,0.3,0.9680961966514587
# 0.45,0.35,0.9661168456077576
# 0.45,0.4,0.9658427834510803
# 0.45,0.45,0.9642968773841858
# 0.45,0.5,0.9619309306144714
# 0.5,0.1,0.9703463912010193
# 0.5,0.15,0.9703463912010193
# 0.5,0.2,0.9699578881263733
# 0.5,0.25,0.9699578881263733
# 0.5,0.3,0.9680914282798767
# 0.5,0.35,0.9661144614219666
# 0.5,0.4,0.9658427834510803
# 0.5,0.45,0.9642968773841858
# 0.5,0.5,0.9619309306144714


################################################
## WBF
################################################
# iou_thresh,score_thresh,mAP@0.5:0.75
# 0.1,0.1,0.9697034358978271
# 0.1,0.15,0.9718723893165588
# 0.1,0.2,0.9712323546409607
# 0.1,0.25,0.9717895984649658
# 0.1,0.3,0.9707955718040466
# 0.1,0.35,0.9694214463233948
# 0.1,0.4,0.9690998196601868
# 0.1,0.45,0.9657287001609802
# 0.1,0.5,0.9639574885368347
# 0.15,0.1,0.9697034358978271
# 0.15,0.15,0.9718723893165588
# 0.15,0.2,0.9712323546409607
# 0.15,0.25,0.9717895984649658
# 0.15,0.3,0.9707955718040466
# 0.15,0.35,0.9694214463233948
# 0.15,0.4,0.9690998196601868
# 0.15,0.45,0.9657287001609802
# 0.15,0.5,0.9639574885368347
# 0.2,0.1,0.9697034358978271
# 0.2,0.15,0.9718723893165588
# 0.2,0.2,0.9712323546409607
# 0.2,0.25,0.9717895984649658
# 0.2,0.3,0.9707955718040466
# 0.2,0.35,0.9694214463233948
# 0.2,0.4,0.9690998196601868
# 0.2,0.45,0.9657287001609802
# 0.2,0.5,0.9639574885368347
# 0.25,0.1,0.9697034358978271
# 0.25,0.15,0.9718723893165588
# 0.25,0.2,0.9712323546409607
# 0.25,0.25,0.9717895984649658
# 0.25,0.3,0.9707955718040466
# 0.25,0.35,0.9694214463233948
# 0.25,0.4,0.9690998196601868
# 0.25,0.45,0.9657287001609802
# 0.25,0.5,0.9639574885368347
# 0.3,0.1,0.9697034358978271
# 0.3,0.15,0.9708917737007141
# 0.3,0.2,0.9712323546409607
# 0.3,0.25,0.9717895984649658
# 0.3,0.3,0.9707955718040466
# 0.3,0.35,0.9694214463233948
# 0.3,0.4,0.9690998196601868
# 0.3,0.45,0.9657287001609802
# 0.3,0.5,0.9639574885368347
# 0.35,0.1,0.9697089791297913
# 0.35,0.15,0.9709010720252991
# 0.35,0.2,0.9712336659431458
# 0.35,0.25,0.9717895984649658
# 0.35,0.3,0.9707955718040466
# 0.35,0.35,0.9694214463233948
# 0.35,0.4,0.9690998196601868
# 0.35,0.45,0.9657287001609802
# 0.35,0.5,0.9639574885368347
# 0.4,0.1,0.9697089791297913
# 0.4,0.15,0.9709010720252991
# 0.4,0.2,0.9712336659431458
# 0.4,0.25,0.9717895984649658
# 0.4,0.3,0.9707955718040466
# 0.4,0.35,0.9694214463233948
# 0.4,0.4,0.9690998196601868
# 0.4,0.45,0.9657287001609802
# 0.4,0.5,0.9639574885368347
# 0.45,0.1,0.969697892665863
# 0.45,0.15,0.9709005355834961
# 0.45,0.2,0.97123783826828
# 0.45,0.25,0.9717950224876404
# 0.45,0.3,0.9707998633384705
# 0.45,0.35,0.9694214463233948
# 0.45,0.4,0.9690998196601868
# 0.45,0.45,0.9657287001609802
# 0.45,0.5,0.9639574885368347
# 0.5,0.1,0.9702332019805908
# 0.5,0.15,0.9712006449699402
# 0.5,0.2,0.9712073802947998
# 0.5,0.25,0.9715803265571594
# 0.5,0.3,0.9706361293792725
# 0.5,0.35,0.9691805839538574
# 0.5,0.4,0.9688961505889893
# 0.5,0.45,0.9652941823005676
# 0.5,0.5,0.96356600522995


################################################
## DIoU + TTA
################################################

# iou_thresh,score_thresh,mAP@0.5:0.75
# 0.1,0.1,0.9697441458702087
# 0.1,0.15,0.9697441458702087
# 0.1,0.2,0.9697441458702087
# 0.1,0.25,0.9697441458702087
# 0.1,0.3,0.9697441458702087
# 0.1,0.35,0.9697441458702087
# 0.1,0.4,0.9697441458702087
# 0.1,0.45,0.9697441458702087
# 0.1,0.5,0.9697441458702087
# 0.15,0.1,0.9697427153587341
# 0.15,0.15,0.9697427153587341
# 0.15,0.2,0.9697427153587341
# 0.15,0.25,0.9697427153587341
# 0.15,0.3,0.9697427153587341
# 0.15,0.35,0.9697427153587341
# 0.15,0.4,0.9697427153587341
# 0.15,0.45,0.9697427153587341
# 0.15,0.5,0.9697427153587341
# 0.2,0.1,0.9697427153587341
# 0.2,0.15,0.9697427153587341
# 0.2,0.2,0.9697427153587341
# 0.2,0.25,0.9697427153587341
# 0.2,0.3,0.9697427153587341
# 0.2,0.35,0.9697427153587341
# 0.2,0.4,0.9697427153587341
# 0.2,0.45,0.9697427153587341
# 0.2,0.5,0.9697427153587341
# 0.25,0.1,0.9697427153587341
# 0.25,0.15,0.9697427153587341
# 0.25,0.2,0.9697427153587341
# 0.25,0.25,0.9697427153587341
# 0.25,0.3,0.9697427153587341
# 0.25,0.35,0.9697427153587341
# 0.25,0.4,0.9697427153587341
# 0.25,0.45,0.9697427153587341
# 0.25,0.5,0.9697427153587341
# 0.3,0.1,0.9697363376617432
# 0.3,0.15,0.9697363376617432
# 0.3,0.2,0.9697363376617432
# 0.3,0.25,0.9697363376617432
# 0.3,0.3,0.9697363376617432
# 0.3,0.35,0.9697363376617432
# 0.3,0.4,0.9697363376617432
# 0.3,0.45,0.9697363376617432
# 0.3,0.5,0.9697363376617432
# 0.35,0.1,0.9697363376617432
# 0.35,0.15,0.9697363376617432
# 0.35,0.2,0.9697363376617432
# 0.35,0.25,0.9697363376617432
# 0.35,0.3,0.9697363376617432
# 0.35,0.35,0.9697363376617432
# 0.35,0.4,0.9697363376617432
# 0.35,0.45,0.9697363376617432
# 0.35,0.5,0.9697363376617432
# 0.4,0.1,0.9697363376617432
# 0.4,0.15,0.9697363376617432
# 0.4,0.2,0.9697363376617432
# 0.4,0.25,0.9697363376617432
# 0.4,0.3,0.9697363376617432
# 0.4,0.35,0.9697363376617432
# 0.4,0.4,0.9697363376617432
# 0.4,0.45,0.9697363376617432
# 0.4,0.5,0.9697363376617432
# 0.45,0.1,0.9700061678886414
# 0.45,0.15,0.9700061678886414
# 0.45,0.2,0.9700061678886414
# 0.45,0.25,0.9700061678886414
# 0.45,0.3,0.9700061678886414
# 0.45,0.35,0.9696640372276306
# 0.45,0.4,0.9696640372276306
# 0.45,0.45,0.9696640372276306
# 0.45,0.5,0.9696640372276306
# 0.5,0.1,0.9700602889060974
# 0.5,0.15,0.9699594378471375
# 0.5,0.2,0.9698143601417542
# 0.5,0.25,0.9698143601417542
# 0.5,0.3,0.9698143601417542
# 0.5,0.35,0.9694721698760986
# 0.5,0.4,0.9694721698760986
# 0.5,0.45,0.9694721698760986
# 0.5,0.5,0.9694721698760986


################################################
## WBF + TTA
################################################
# iou_thresh,score_thresh,mAP@0.5:0.75
# 0.1,0.1,0.9838979244232178
# 0.1,0.15,0.9833899140357971
# 0.1,0.2,0.9836897253990173
# 0.1,0.25,0.9842371940612793
# 0.1,0.3,0.9841075539588928
# 0.1,0.35,0.9838888049125671
# 0.1,0.4,0.9836196899414062
# 0.1,0.45,0.9824966788291931
# 0.1,0.5,0.9819128513336182
# 0.15,0.1,0.9838979244232178
# 0.15,0.15,0.9833899140357971
# 0.15,0.2,0.9836897253990173
# 0.15,0.25,0.9842371940612793
# 0.15,0.3,0.9841075539588928
# 0.15,0.35,0.9838888049125671
# 0.15,0.4,0.9836196899414062
# 0.15,0.45,0.9824966788291931
# 0.15,0.5,0.9819128513336182
# 0.2,0.1,0.9838977456092834
# 0.2,0.15,0.9833879470825195
# 0.2,0.2,0.9836878776550293
# 0.2,0.25,0.9842371940612793
# 0.2,0.3,0.9841063618659973
# 0.2,0.35,0.9838888049125671
# 0.2,0.4,0.9836196899414062
# 0.2,0.45,0.9824966788291931
# 0.2,0.5,0.9819128513336182
# 0.25,0.1,0.9847074151039124
# 0.25,0.15,0.9833979606628418
# 0.25,0.2,0.9836926460266113
# 0.25,0.25,0.983997642993927
# 0.25,0.3,0.9841063618659973
# 0.25,0.35,0.9838888049125671
# 0.25,0.4,0.9836196899414062
# 0.25,0.45,0.9824966788291931
# 0.25,0.5,0.9819128513336182
# 0.3,0.1,0.9846575856208801
# 0.3,0.15,0.9833466410636902
# 0.3,0.2,0.9836950302124023
# 0.3,0.25,0.9840002655982971
# 0.3,0.3,0.9841086268424988
# 0.3,0.35,0.9838888049125671
# 0.3,0.4,0.9836196899414062
# 0.3,0.45,0.9824972748756409
# 0.3,0.5,0.9819130897521973
# 0.35,0.1,0.9846575856208801
# 0.35,0.15,0.9833466410636902
# 0.35,0.2,0.9836950302124023
# 0.35,0.25,0.9840002655982971
# 0.35,0.3,0.9841086268424988
# 0.35,0.35,0.9838888049125671
# 0.35,0.4,0.9836196899414062
# 0.35,0.45,0.9824972748756409
# 0.35,0.5,0.9819130897521973
# 0.4,0.1,0.9846559166908264
# 0.4,0.15,0.9833500385284424
# 0.4,0.2,0.9836950302124023
# 0.4,0.25,0.9840012192726135
# 0.4,0.3,0.9841094613075256
# 0.4,0.35,0.9838873744010925
# 0.4,0.4,0.9836196899414062
# 0.4,0.45,0.9824957847595215
# 0.4,0.5,0.9819116592407227
# 0.45,0.1,0.9854231476783752
# 0.45,0.15,0.9836974143981934
# 0.45,0.2,0.984059751033783
# 0.45,0.25,0.9840021133422852
# 0.45,0.3,0.9841108322143555
# 0.45,0.35,0.9838890433311462
# 0.45,0.4,0.9836180806159973
# 0.45,0.45,0.9824957847595215
# 0.45,0.5,0.9819116592407227
# 0.5,0.1,0.9852632880210876
# 0.5,0.15,0.9834897518157959
# 0.5,0.2,0.9843370318412781
# 0.5,0.25,0.9838964343070984
# 0.5,0.3,0.9839127063751221
# 0.5,0.35,0.9830084443092346
# 0.5,0.4,0.9826939702033997
# 0.5,0.45,0.9814903736114502
# 0.5,0.5,0.9808415770530701

################################################
## WBF + TTA + Neighbors
################################################
# n = 1:
# n = 2:
# n = 3:
# n = 4:
# n = 5:
# n = 6:
# n = 7:
# n = 8:
