#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2 as cv
import tensorflow as tf
import os
from tabulate import tabulate
import numpy as np
import time
import albumentations as A

tf.get_logger().setLevel("INFO")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
from yolov4.common.predict import DIoU_NMS

from config import config
import utils
from utils import YoloBBox, MeanAveragePrecision, special_nms

import click


def pbench(t, name):
    print(f"{name}::{time.perf_counter() - t:.4}s")


check_errors = True


@click.command()
@click.argument("dataset", type=click.STRING)
@click.option("-i", "--iou_thresh", type=click.FLOAT, default=0.3)
@click.option("-s", "--score_thresh", type=click.FLOAT, default=0.25)
@click.option("-t", "--text_thresh", type=click.FLOAT, default=0.3)
@click.option("-o", "--occlusion_thresh", type=click.FLOAT, default=0.3)
@click.option("-v", "--vote_thresh", type=click.INT, default=4)
@click.option("--tta", type=click.BOOL, is_flag=True, default=False)
@click.option("--wbf", type=click.BOOL, is_flag=True, default=False)
@click.option("--show", type=click.BOOL, is_flag=True, default=False)
@click.option("--threshold_tuning", type=click.BOOL, is_flag=True, default=False)
@click.option("--snms", type=click.BOOL, is_flag=True, default=False)
@click.option("--sub", type=click.BOOL, is_flag=True, default=False)
@click.option("--diou", type=click.BOOL, is_flag=True, default=False)
def main(
    dataset,
    iou_thresh,
    score_thresh,
    text_thresh,
    occlusion_thresh,
    vote_thresh,
    tta,
    wbf,
    show,
    threshold_tuning,
    snms,
    sub,
    diou,
):
    #### PARAMS ####
    if dataset == "train":
        dir_ = config.train_out_dir
    elif dataset == "valid":
        dir_ = config.valid_out_dir
    elif dataset == "test":
        dir_ = config.test_out_dir
    else:
        raise ValueError(f"Dataset '{dataset}' invalid.")

    wbf = True if wbf else False
    tta = True if tta else False
    show = True if show else False
    threshold_tuning = True if threshold_tuning else False
    snms = True if snms else False
    sub = True if sub else False
    diou = True if diou else False

    #### THRESHOLD TUNING #####
    threshold_params = []
    if threshold_tuning:
        # fmt: off
        thresholds = []
        # for iou in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        #     for score in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        for iou in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
            for score in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
                thresholds.append((iou, score))
        # fmt: on
        mAPs_for_thresholds = []
    else:
        thresholds = [[iou_thresh, score_thresh]]

    for iou_thresh, score_thresh in thresholds:
        for input_size in (config.yolo.input_size,):  # 832):
            ##################
            ### INIT YOLO ####
            ##################
            yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
            yolo.classes = config.yolo.classes
            yolo.input_size = input_size
            yolo.channels = config.yolo.channels
            yolo.make_model()
            yolo.load_weights(
                config.yolo.weights, weights_type=config.yolo.weights_type
            )

            for idx, cls_name in yolo.classes.items():
                if cls_name == "text":
                    text_label_idx = idx
                    break

            ##################
            ### PARAMTER  ####
            ##################
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")
            print("Running mAP calculation with:")
            # fmt: off
            pretty = [["Dataset", "Input", "IoU-Thresh", "Score-Thresh", "TTA-Enabled", "NMS"]]
            pretty += [[dataset, input_size, iou_thresh, score_thresh, True if tta else False, "WBF" if wbf else "DIoU"]]
            pretty += [["Weights", config.yolo.weights]]
            print(tabulate(pretty))
            # fmt: on

            if tta:
                yolo_tta = utils.YoloTTA(
                    yolo.classes,
                    config.augment.label_transition_flip,
                    config.augment.label_transition_rotation,
                )

            preds, gts, original_gts, original_shapes, orignal_imgs = [], [], [], [], []
            for img_path in utils.list_imgs(dir_):
                # for img_path in (Path("data/valid_out/00_22_a_000_nflip_aug.jpg"),):
                # for img_path in (Path("data/test_out/08_07_c_000_nflip_aug.png"),):
                print(img_path)

                def test_augmentations(img, bboxes):
                    resize = A.Compose(
                        [
                            A.LongestMaxSize(
                                max_size=input_size,
                                always_apply=True,
                            )
                        ],
                        bbox_params=A.BboxParams("yolo"),
                    )
                    pad = A.Compose(
                        [
                            A.PadIfNeeded(
                                min_height=input_size,
                                min_width=input_size,
                                border_mode=cv.BORDER_CONSTANT,
                                value=0,
                                always_apply=True,
                            )
                        ],
                        bbox_params=A.BboxParams("yolo"),
                    )
                    gt = bboxes.copy()

                    bboxes = utils.A.class_to_back(bboxes)

                    transform = resize(image=img, bboxes=bboxes)
                    img, bboxes = transform["image"], transform["bboxes"]

                    # DEBUG
                    # print("After resize")
                    # bboxes = utils.A.class_to_front(bboxes)
                    # utils.show_bboxes(img, bboxes, type_="gt", gt=gt)
                    # bboxes = utils.A.class_to_back(bboxes)

                    resized = img.copy()
                    transform = pad(image=img, bboxes=bboxes)
                    img, bboxes = transform["image"], transform["bboxes"]

                    # DEBUG
                    # print("After pad")
                    # gt = utils.A.class_to_back(gt)
                    # _, gt = yolo.resize_image(resized, gt)
                    # gt = utils.A.class_to_front(gt)
                    # bboxes = utils.A.class_to_front(bboxes)
                    # utils.show_bboxes(img, bboxes, type_="gt", gt=gt)
                    # bboxes = utils.A.class_to_back(bboxes)

                    bboxes = utils.A.class_to_front(bboxes)

                    return img, bboxes, resized

                img = np.expand_dims(
                    cv.imread(str(img_path), cv.IMREAD_GRAYSCALE), axis=2
                )
                img = utils.resize_max_axis(img, 1000)
                orig = img.copy()

                gt_path = utils.Yolo.label_from_img(img_path)
                gt = utils.load_ground_truth(gt_path)

                original_gts.append(gt)
                orignal_imgs.append(orig)
                original_shapes.append(orig.shape)

                img, gt, resized_img = test_augmentations(img, gt)
                if show:
                    utils.show(img)

                gts.append(gt)

                if wbf:
                    wbf_nms = utils.WBF(
                        iou_thresh=iou_thresh,
                        score_thresh=score_thresh,
                        vote_thresh=vote_thresh if tta else 1,
                        conf_type="avg",  # avg, max
                    )

                ######################
                ### TTA PREDICTION ###
                ######################

                # will do wbf two times, after each model prediction and then again
                # on all final predictions
                if tta and wbf and sub:
                    # create all img augmentations for ensemble
                    imgs_combs = yolo_tta.augment(img.copy())
                    # store the tta_preds
                    tta_preds = []
                    for aug_img, comb in imgs_combs:

                        # if raw is set no nms is applied, but boxes based on score are
                        # still removed
                        pred = yolo.predict(
                            aug_img,
                            iou_threshold=iou_thresh,
                            score_threshold=score_thresh,
                            raw=True,
                        )

                        if diou:
                            pred = DIoU_NMS(pred, threshold=iou_thresh)
                        else:
                            inner_wbf_nms = utils.WBF(
                                iou_thresh=iou_thresh,
                                score_thresh=score_thresh,
                                vote_thresh=1,
                                conf_type="avg",  # avg, max
                            )
                            pred = inner_wbf_nms.perform([pred])

                        # de augment the prediction
                        pred = yolo_tta.perform(pred, comb)

                        # DEBUG
                        # utils.show_bboxes(img, pred, type_="pred", classes=yolo.classes)

                        tta_preds.append(pred)

                    if show:
                        print("Before WBF")
                        utils.show_bboxes(
                            img,
                            np.vstack(tta_preds),
                            type_="pred",
                            classes=yolo.classes,
                        )

                    pred = wbf_nms.perform(tta_preds)
                    if snms:
                        # pred = special_nms.fuse_textboxes(
                        #     pred, text_label_idx, text_thresh
                        # )
                        pred = special_nms.remove_occlusion(pred, occlusion_thresh)

                    if show:
                        print("Prediction")
                        utils.show_bboxes(img, pred, type_="pred", classes=yolo.classes)

                    preds.append(pred)

                ########################################################################
                ### TTA PREDICTION #####################################################
                ########################################################################
                elif tta:
                    # create all img augmentations for ensemble
                    # TODO other combs only 180 and flip
                    # TODO arbeit
                    imgs_combs = yolo_tta.augment(img.copy(), times=8)
                    # store the tta_preds
                    tta_preds = []
                    for aug_img, comb in imgs_combs:
                        # DEBUG
                        # utils.show(aug_img)

                        # if raw is set no nms is applied, but boxes based on score are
                        # still removed
                        pred = yolo.predict(
                            aug_img,
                            iou_threshold=iou_thresh,
                            score_threshold=score_thresh,
                            raw=True,
                        )
                        # de augment the prediction
                        pred = yolo_tta.perform(pred, comb)
                        tta_preds.append(pred)

                        # DEBUG
                        # utils.show_bboxes(img, pred, type_="pred")

                    # DEBUG: show all predicted bboxes
                    if show:
                        utils.show_bboxes(img, np.vstack(tta_preds), type_="pred")

                    if wbf:
                        pred = wbf_nms.perform(tta_preds)
                        pred = np.vstack(pred)

                        # if show:
                        #     print("BeforeTextFuse")
                        #     utils.show_bboxes(
                        #         img, pred, type_="pred", classes=yolo.classes
                        #     )
                        if snms:
                            pred = special_nms.fuse_textboxes(
                                pred, text_label_idx, text_thresh
                            )

                        # if show:
                        #     print("BeforeOcclusion")
                        #     utils.show_bboxes(
                        #         img, pred, type_="pred", classes=yolo.classes
                        #     )
                        if snms:
                            pred = special_nms.remove_occlusion(pred, occlusion_thresh)

                    else:
                        pred = DIoU_NMS(np.vstack(tta_preds), threshold=iou_thresh)

                    # DEBUG: show the tta pred
                    if show:
                        print("Prediction")
                        utils.show_bboxes(img, pred, type_="pred", classes=yolo.classes)
                        print("GroundTruth")
                        utils.show_bboxes(img, gt, type_="gt", classes=yolo.classes)

                    preds.append(pred)

                ########################################################################
                ### NORMAL PREDICTION ##################################################
                ########################################################################
                else:
                    pred = yolo.predict(
                        img,
                        iou_threshold=iou_thresh,
                        score_threshold=score_thresh,
                        raw=True,
                    )
                    if wbf:
                        pred = wbf_nms.perform([pred])
                    else:
                        pred = DIoU_NMS(pred, threshold=iou_thresh)

                    if snms:
                        pred = special_nms.fuse_textboxes(
                            pred, text_label_idx, text_thresh
                        )
                        pred = special_nms.remove_occlusion(pred, occlusion_thresh)

                    # if show:
                    #     print("After NMS.")
                    #     utils.show_bboxes(padded_img, pred, type_="pred", classes=yolo.classes)

                    if show:
                        # ONLY TEST
                        # pred = np.delete(pred, [0], axis=0)
                        print("Prediction")
                        utils.show_bboxes(
                            img, pred, type_="pred", classes=yolo.classes, gt=gt
                        )
                        utils.show_bboxes(
                            resized_img,
                            yolo.fit_pred_bboxes_to_original(pred, resized_img.shape),
                            type_="pred",
                            classes=yolo.classes,
                            gt=original_gts[-1],
                        )
                        # for p in pred:
                        #     utils.show_bboxes(img, [p], type_="pred", classes=yolo.classes)

                    if show:
                        print("Ground truth.")
                        utils.show_bboxes(img, gt, type_="gt", classes=yolo.classes)

                    preds.append(pred)

            ############################################################################
            ######################## MAP ###############################################
            ############################################################################
            iou_threshs = (0.5, 0.76, 0.05)
            # iou_threshs = (0.5, 1.0, 0.05)

            # absolute = False
            absolute = True

            ################################
            ### ORIGINAL SIZE ##############
            ################################

            original_preds = [
                yolo.fit_pred_bboxes_to_original(pred, original_shape)
                for pred, original_shape in zip(preds, original_shapes)
            ]

            # DEBUG check fitted stuff
            for pred in original_preds:
                for fitted_box in pred:
                    bbox = fitted_box[:4]

                    if not np.all(np.logical_and(0 < bbox,  bbox < 1)):
                        assert False, "0 < bbox < 1"


            mAP = utils.MeanAveragePrecision(
                yolo.classes,
                original_shapes,
                iou_threshs=(0.5, iou_threshs),
                # policy="soft",
                policy="greedy",
                same_img_shape=False,
            )
            mAP.add(original_preds, original_gts, absolute=absolute)
            results = mAP.compute()
            pretty = mAP.prettify(results)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!! MAP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(pretty)

            # DEBUG
            # for oimg, gt, pred in zip(orignal_imgs, original_gts, original_preds):
            #     utils.show_bboxes(oimg, pred, type_="pred", classes=yolo.classes)
            #     utils.show_bboxes(oimg, gt, type_="gt", classes=yolo.classes)

            ############################################################################
            ######################## MY ################################################
            ############################################################################

            metric_result = []
            metrics = ["f1", "recall", "precision"]
            for iou in np.arange(*iou_threshs):
                # my metrics
                calculator = utils.Metrics(yolo.classes, dir_, iou_thresh=iou)
                for gt, pred, shape in zip(
                    original_gts, original_preds, original_shapes
                ):
                    calculator.calculate(gt, pred, shape)

                res = calculator.perform(metrics)
                metric_result.append(res)

                # calculator.prettify(metrics, res, iou=iou)

            metric_result = np.array(metric_result).mean(axis=0)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!! ME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            calculator.prettify(metrics, metric_result, iou="0.5:0.75")

    if threshold_tuning:
        print("iou_thresh,score_thresh,mAP@0.5:0.75")
        for iou_thresh, score_thresh, mAP in mAPs_for_thresholds:
            print(f"{iou_thresh},{score_thresh},{mAP}")

            # bbox_gt = [YoloBBox(img.shape).from_ground_truth(gt) for gt in ground_truth]
            # bbox_gt = np.vstack([[*bbox.abs(), bbox.label, 0, 0] for bbox in bbox_gt])

            # print(bbox_gt.shape)

            # bbox_pred = [
            #     YoloBBox(img.shape).from_prediction(pred) for pred in predictions
            # ]
            # bbox_pred = np.vstack(
            #     [[*bbox.abs(), bbox.label, bbox.confidence] for bbox in bbox_pred]
            # )
            # # print(bbox_pred.shape)

            # mAP.add(bbox_pred, bbox_gt)

            # iou_threshs = np.arange(0.5, 1, 0.05)
            # results_095 = mAP.value(
            #     iou_thresholds=iou_threshs, recall_thresholds=None, mpolicy="greedy"
            # )
            # results_05 = mAP.value(iou_thresholds=0.5, recall_thresholds=None, mpolicy="greedy")

            # pretty = [["Class", "AP@0.5"]] #, "mAP@0.5", "mAP@0.5:0.95"]]
            # map05 = results_05["mAP"]
            # map095 = results_095["mAP"]
            # print(results_095)

            # results_05 = sorted(results_05[0.5].items())
            # results_095 = sorted(results_095[0.5].items())
            # for (cls_idx, data05), (cls_idx2, data095) in zip(results_05, results_095):
            #     assert cls_idx == cls_idx2

            #     ffloat = lambda f: "{:.3f}".format(f)

            #     cls_name = yolo.classes[cls_idx]
            #     # ap05, ap095  = data05["ap"], data095["ap"]
            #     ap05 = data05["ap"]
            #     pretty += [[cls_name, ffloat(ap05)]]

            # print(tabulate(pretty))

            # err = metrics.calculate(ground_truth, predictions, img.shape[:2])
            # errors.append((img, img_path, *err))

            # metrics.confusion()
            # metrics.perform(["f1", "precision", "recall"], precision=4)

            # if check_errors:
            #     has_errors = lambda x, y, z: (x or y or z)
            #     for img, img_path, wrong_prediction, unmatched_gt, unmatched_pred in errors:
            #         if not has_errors(wrong_prediction, unmatched_gt, unmatched_pred):
            #             continue

            #         print("Image:", img_path)
            #         for err_bbox in wrong_prediction:
            #             print("Wrong predictions")
            #             bbox = np.array([err_bbox.yolo()])
            #             tmp = img.copy()
            #             tmp = yolo.draw_bboxes(tmp, bbox)
            #             utils.show(tmp)

            #         # TODO
            #         for err_bbox in unmatched_gt:
            #             print("Unmatched Ground truth")
            #             bbox = np.array([err_bbox.yolo()])
            #             tmp = img.copy()
            #             tmp = yolo.draw_bboxes(tmp, bbox)
            #             utils.show(tmp)

            #         for err_bbox in unmatched_pred:
            #             print("Unmatched Prediction")
            #             bbox = np.array([err_bbox.yolo()])
            #             tmp = img.copy()
            #             tmp = yolo.draw_bboxes(tmp, bbox)
            #             utils.show(tmp)


#########################################
############### OLD #####################
#########################################

# ################################
# ### CONSTANT SIZE ##############
# ################################

# shape = (input_size, input_size)
# mAP = utils.MeanAveragePrecision(
#     yolo.classes,
#     shape,
#     iou_threshs=(0.5, iou_threshs),
#     # policy="soft",
#     policy="greedy",
# )
# mAP.add(preds, gts, absolute=absolute)
# results = mAP.compute()
# pretty = mAP.prettify(results)
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!! MAP PADDED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(pretty)

# mAPs = mAP.get_maps(results)
# relevant_mAP = mAPs[-1]
# if threshold_tuning:
#     mAPs_for_thresholds.append((iou_thresh, score_thresh, relevant_mAP))

# ###################################
# ####### PADDED SIZE #############
# ###################################
# metric_result = []
# metrics = ["f1", "recall", "precision"]
# for iou in np.arange(*iou_threshs):
#     # my metrics
#     calculator = utils.Metrics(yolo.classes, dir_, iou_thresh=iou)
#     for gt, pred in zip(gts, preds):
#         calculator.calculate(gt, pred, (input_size, input_size))

#     res = calculator.perform(metrics)
#     metric_result.append(res)

#     # calculator.prettify(metrics, res, iou=iou)

# metric_result = np.array(metric_result).mean(axis=0)
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!! MY PADDED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# calculator.prettify(metrics, metric_result, iou="0.5:0.75")
if __name__ == "__main__":
    main()
