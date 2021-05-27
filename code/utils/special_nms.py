import numba as nb
import numpy as np

import utils

@nb.njit
def delete_row(arr, num):
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[np.where(arr == num)[0]] = False
    return arr[mask]

# @nb.njit
def fuse_textboxes(pred, text_label_idx, fuse_textbox_iou=0.5):
    for idx1 in range(0, len(pred) - 1):
        for idx2 in range(idx1 + 1, len(pred)):
            bbox1 = pred[idx1]
            bbox2 = pred[idx2]

            # print(bbox1)
            # print(bbox2)
            bbox1_label = bbox1[4]
            bbox2_label = bbox2[4]

            # if one is not a text
            if bbox1_label != text_label_idx or bbox2_label != text_label_idx:
                continue

            if utils.nb_calc_iou_yolo(bbox1, bbox2) < fuse_textbox_iou:
                continue

            idx1, idx2 = sorted((idx1, idx2))
            pred = np.delete(pred, (idx1, idx2), axis=0)
            # pred = delete_row(pred, idx2)
            # pred = delete_row(pred, idx1)

            # convert to x1,y2,x2,y2 format
            (
                bbox1_x1,
                bbox1_y1,
                bbox1_x2,
                bbox1_y2,
                bbox1_l,
                bbox1_c,
            ) = utils.nb_convert_from_yolo_bbox(bbox1)

            (
                bbox2_x1,
                bbox2_y1,
                bbox2_x2,
                bbox2_y2,
                bbox2_l,
                bbox2_c,
            ) = utils.nb_convert_from_yolo_bbox(bbox2)

            new_x1, new_y1 = min(bbox1_x1, bbox2_x1), min(bbox1_y1, bbox2_y1)
            new_x2, new_y2 = max(bbox1_x2, bbox2_x2), max(bbox1_y2, bbox2_y2)
            new_conf = max(bbox1_c, bbox2_c)

            new_bbox = np.array(
                utils.nb_convert_to_yolo_bbox(
                    (new_x1, new_y1, new_x2, new_y2, bbox1_l, new_conf)
                )
            ).reshape(1, -1)

            # print("del1", bbox1)
            # print("del2", bbox2)
            # print("fuse", new_bbox)

            pred = np.append(pred, new_bbox, axis=0)
            return fuse_textboxes(pred, text_label_idx, fuse_textbox_iou)

    return pred

def remove_occlusion(pred, occlusion_iou=0.3):
    bboxes_to_remove = []
    for idx1 in range(len(pred) - 1):
        for idx2 in range(idx1 + 1, len(pred)):
            bbox1 = pred[idx1]
            bbox2 = pred[idx2]

            # when a certain thresh is reached between both
            if utils.nb_calc_iou_yolo(bbox1, bbox2) < occlusion_iou:
                continue

            bbox1_conf = bbox1[5]
            bbox2_conf = bbox2[5]

            # remove the one with the lower confidence
            if bbox1_conf < bbox2_conf:
                bboxes_to_remove.append(idx1)
            else:
                bboxes_to_remove.append(idx2)

    pred = np.delete(pred, bboxes_to_remove, axis=0)

    return pred
