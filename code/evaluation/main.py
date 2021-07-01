import cv2 as cv
import numpy as np
from pathlib import Path

from config import config
import utils
from evaluation import (
    TopologyEvaluator,
    ArrowAndTextMatchingEvaluator,
    GroundTruth,
    Prediction,
    Result,
)
from pipeline import pipeline
from tabulate import tabulate

import click


def p(*args, **kwargs):
    print("----------------------")
    print(*args, **kwargs)


def pm(name, m):
    print("----------------------")
    print(name)

    legend = []
    for i in range(10):
        legend.extend((i, i))

    print(np.array(legend).reshape(1, -1)[0])
    for r in m:
        print(r)


def recall(res):
    return res.TPS / (res.TPS + res.FNS)


def precision(res):
    return res.TPS / (res.TPS + res.FPS)


def f1(res):
    r = recall(res)
    p = precision(res)

    return 2 * p * r / (p + r)


def metrics(res):
    return precision(res), recall(res), f1(res)


def ffloat(f):
    return f"{f*100:.4}%"


@click.command()
# @click.argument("dataset")
@click.option("-ys", "--yolo_score_thresh", type=float, default=0.5)
@click.option("-yi", "--yolo_iou_thresh", type=float, default=0.3)
@click.option("-yin", "--yolo_input_size", type=int, default=608)
@click.option("-ynms", "--yolo_nms", type=str, default="diou")
@click.option("-yvote", "--yolo_vote_thresh", type=int, default=1)
@click.option("--ytta", is_flag=True, default=False)
@click.option("--yload", is_flag=True, default=False)
@click.option("-uin", "--unet_input_size", type=int, default=448)
@click.option("--show", is_flag=True, default=False)
def main(
    yolo_score_thresh,
    yolo_iou_thresh,
    yolo_input_size,
    ytta,
    yolo_nms,
    yolo_vote_thresh,
    yload,
    unet_input_size,
    show,
):
    yolo_tta = ytta
    import time

    tstart = time.perf_counter()


    yolo_weights = config.yolo.weights

    all_unet_weights = [
        'experiments_unet/grid/grid_bs_32_loss_focal2_0.1_lr_0.01/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_focal2_0.1_lr_0.0025/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_focal2_0.1_lr_0.0001/run2/best.pth',

        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.1_lr_0.01/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.1_lr_0.005/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.1_lr_0.0025/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.1_lr_0.0001/run2/best.pth',

        # 'experiments_unet/grid/grid_bs_32_loss_focal2_0.8_lr_0.005/run2/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_focal2_0.8_lr_0.0025/run2/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_focal2_0.8_lr_0.00025/run2/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_focal2_0.8_lr_0.0001/run0/best.pth',

        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.8_lr_0.01/run1/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.8_lr_0.001/run2/best.pth',
        # "experiments_unet/grid/grid_bs_64_loss_focal2_0.8_lr_0.0005/run2/best.pth",
        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.8_lr_0.0001/run2/best.pth',

        # 'experiments_unet/grid/grid_bs_32_loss_dice_lr_0.01/run2/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_dice_lr_0.005/run1/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_dice_lr_0.001/run1/best.pth',
        # 'experiments_unet/grid/grid_bs_32_loss_dice_lr_0.0001/run2/best.pth',

        # 'experiments_unet/grid/grid_bs_64_loss_dice_lr_0.001/run1/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_dice_lr_0.005/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_dice_lr_0.0025/run2/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_dice_lr_0.0001/run2/best.pth',

        # 'experiments_unet/grid/grid_bs_64_loss_dice_lr_0.001/run0/best.pth',
        # 'experiments_unet/grid/grid_bs_64_loss_focal2_0.8_lr_0.01/run0/best.pth',
    ]
    # unet_weights = unet_weights[0]

    img_paths = [
        "data/eval/07_00_eval.png",
        "data/eval/07_01_eval.png",
        "data/eval/07_02_eval.png",
        "data/eval/07_03_eval.png",
        "data/eval/07_04_eval.png",
        "data/eval/07_05_c_eval.png",
        "data/eval/07_06_c_eval.png",
        "data/eval/07_07_c_eval.png",
        "data/eval/07_08_c_eval.png",
        "data/eval/07_09_c_a_eval.png",
        "data/eval/07_10_c_a_eval.png",
        "data/eval/07_11_c_a_eval.png",
        "data/eval/07_12_c_a_eval.png",
        "data/eval/07_13_c_a_eval.png",
        "data/eval/08_00_eval.png",
        "data/eval/08_01_eval.png",
        "data/eval/08_02_eval.png",
        "data/eval/08_03_eval.png",
        "data/eval/08_04_eval.png",
        "data/eval/08_05_eval.png",
        "data/eval/08_06_eval.png",
        "data/eval/08_07_c_eval.png",
        "data/eval/08_08_c_eval.png",
        "data/eval/08_09_c_eval.png",
        "data/eval/08_10_c_eval.png",
        "data/eval/08_11_c_a_eval.png",
        # "data/eval/08_12_c_a_eval.png",  # takes too long
        "data/eval/08_13_c_a_eval.png",
        "data/eval/08_14_c_a_eval.png",
        "data/eval/08_15_c_a_eval.png",
        "data/eval/10_00_eval.png",
        "data/eval/11_00_eval.jpg",
        "data/eval/11_01_eval.jpg",
        "data/eval/11_02_a_eval.jpg",
        "data/eval/11_03_a_eval.jpg",
        "data/eval/11_04_a_eval.jpg",
        "data/eval/11_05_a_eval.jpg",
        "data/eval/20_00_a_eval.jpg",
        "data/eval/20_01_a_eval.jpg",
        "data/eval/20_02_a_eval.jpg",
        "data/eval/20_03_eval.jpg",
        "data/eval/21_00_a_eval.jpg",
        "data/eval/21_01_c_a_eval.jpg",
        "data/eval/21_02_c_a_eval.jpg",
        "data/eval/22_00_eval.jpg",
        "data/eval/22_01_eval.jpg",
        "data/eval/22_02_eval.jpg",
        "data/eval/22_03_eval.jpg",
        "data/eval/22_04_c_eval.jpg",
        "data/eval/22_05_c_eval.jpg",
        "data/eval/22_06_c_eval.jpg",
        "data/eval/25_00_c_a_eval.jpg",
        "data/eval/25_01_c_a_eval.jpg",
        "data/eval/25_02_a_eval.jpg",
        "data/eval/26_01_c_eval.jpg",
        "data/eval/28_00_c_a_eval.png",
    ]

    ##################################################################
    ## CONFIG
    ##################################################################
    use_gt = True

    ##################################################################
    ## CONFIG
    ##################################################################

    for unet_weights in all_unet_weights:
        overall_pred_res = Result()
        overall_topology_res = Result()
        overall_arrows_res = Result()
        overall_texts_res = Result()
        macro_res = Result()

        eval_items = pipeline.predict(
            img_paths,
            yolo_weights=yolo_weights,
            yolo_iou_thresh=yolo_iou_thresh,
            yolo_score_thresh=yolo_score_thresh,
            yolo_nms=yolo_nms,
            yolo_input_size=yolo_input_size,
            yolo_tta=yolo_tta,
            yolo_load=yload,
            yolo_vote_thresh=yolo_vote_thresh,
            unet_input_size=unet_input_size,
            unet_weights=unet_weights,
            use_gt=use_gt,
        )

        exceeded_depths = []
        # imgs = utils.list_imgs(config.eval_dir)
        for img_path, eval_item in zip(img_paths, eval_items):
            print(utils.green("----------------------------------------------------"))
            print(utils.green("----------------------------------------------------"))
            p(str(img_path))

            img_path = Path(img_path)

            img = cv.imread(str(img_path))
            # img = utils.resize_max_axis(img, config.unet.test_input_size)
            img = utils.resize_max_axis(img, yolo_input_size)

            eval_path = config.eval_dir / f"{img_path.stem}.md"

            # DEBUG
            # fmt: off
            if show:
                utils.show_bboxes(img, eval_item.pred_bboxes, type_="utils", classes=eval_item.classes, others=[eval_item.segmentation])
            # fmt: on

            ############################
            ### TOPOLOGY EVALUATION ####
            ############################
            eval_item = pipeline.topology(
                eval_item, fn_threshold=0.3, fuse_textbox_iou=0.2, occlusion_iou=0.5
            )

            gt = GroundTruth(eval_item.gt_bboxes, eval_path)
            pred = Prediction(eval_item.pred_bboxes, eval_item.topology)

            gt_adj, pred_adj = gt.adjacency, pred.adjacency
            evaluator = TopologyEvaluator(gt_adj, pred_adj, eval_item.false_negative_gts)

            try:
                topology_res = evaluator.evaluate()
            except Exception as e:
                print(utils.red("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                print(utils.red("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                print(utils.red("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                print(utils.red("!!!!!!!! ERROR IN TOPOLOGY EVAL !!!!!!!!!!!!"))
                print(utils.red("!!!!!!!!"), e)
                exceeded_depths.append(img_path)
                raise e
                continue

            overall_topology_res += topology_res

            pred_res = eval_item.classification
            overall_pred_res += pred_res

            ############################
            ### MATCHING EVALUATION ####
            ############################
            ### matching
            arrow_and_text_matcher = pipeline.ArrowAndTextMatcher(eval_item)
            eval_item = arrow_and_text_matcher.match(
                distance_algorithm="nearest_neighbor", threshold="TODO"
            )

            ### evaluation
            eval_matching_arrow_path = config.eval_dir / f"{img_path.stem}_arrows.csv"
            eval_matching_text_path = config.eval_dir / f"{img_path.stem}_texts.csv"

            evaluator = ArrowAndTextMatchingEvaluator(
                eval_item, eval_matching_arrow_path, eval_matching_text_path
            )
            arrow_res, text_res = evaluator.evaluate()

            overall_arrows_res += arrow_res
            overall_texts_res += text_res

            #########################
            ### MACRO EVALUATION ####
            #########################
            # if any error occured
            if (
                pred_res.FNS
                or pred_res.FPS
                or topology_res.FNS
                or topology_res.FPS
                or arrow_res.FNS
                or arrow_res.FPS
                or text_res.FNS
                or text_res.FPS
            ):
                macro_res.FNS += 1
            else:
                macro_res.TPS += 1

            ################
            ### RESULTS ####
            ################
            dump_res = lambda x: (x.TPS, x.FPS, x.FNS)
            pretty = [["", "TPs", "FPs", "FNs"]]
            pretty += [["Classification", *dump_res(pred_res)]]
            pretty += [["Topology", *dump_res(topology_res)]]
            pretty += [["Arrow", *dump_res(arrow_res)]]
            pretty += [["Text", *dump_res(text_res)]]
            print(tabulate(pretty))

        print("----------------------")
        print("----------------------")
        # which items exceeded depth
        if exceeded_depths:
            p("Exceeded depth:\n", "\t\n".join(str(path) for path in exceeded_depths))

        ####################################
        ########### RESULTS ################
        ####################################
        tend = time.perf_counter()

        # fmt: off
        dump_res = lambda x: (x.TPS, x.FPS, x.FNS, *(ffloat(v) for v in metrics(x)))
        pretty = [["", "TPs", "FPs", "FNs", "Precision", "Recall", "F1"]]
        pretty += [["Classification", *dump_res(overall_pred_res)]]
        pretty += [["Topology", *dump_res(overall_topology_res)]]
        pretty += [["ArrowMatching", *dump_res(overall_arrows_res)]]
        pretty += [["TextMatching", *dump_res(overall_texts_res)]]
        pretty += [["Macro", macro_res.TPS, macro_res.FPS, macro_res.FNS, "N.A.", ffloat(recall(macro_res)), "N.A."]]
        pretty += [[""]]
        pretty += [["Took", f"{tend - tstart:.3f}s"]]
        # print(tabulate(pretty))
        pretty = tabulate(pretty)
        print(pretty)

        with open("evaluation_results.txt", "a") as f:
            print(pretty, file=f)
            print("")
        # fmt: on


if __name__ == "__main__":
    main()
