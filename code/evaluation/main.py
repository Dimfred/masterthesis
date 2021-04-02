import cv2 as cv
import numpy as np

from config import config
import utils
from evaluation import Evaluator, GroundTruth, Prediction, Result
from pipeline import pipeline


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

def main():
    overall_res = Result()
    macro_res = Result()
    for name in [
        "07_00",
        "07_01",
        "07_02",
        "07_03",
        "07_04",
        "07_05",
        "07_06",
        "07_07",
        "07_08",
        "08_00",
        "08_01",
        "08_02",
        "08_03",
        "08_04",
        "08_05",
        "08_06",
        # TODO maybe do something in the evaluation step
        "08_07", # works when dilation is set to 0 but that won't close all holes
        "08_08",
        "08_09",
        "08_10",
        "10_00",
        "15_00",
        "15_01",
        "15_02",
        "15_03",
    ]:
        p(name)
        name = f"{name}_000_nflip_aug_eval"
        if not name.startswith("15_"):
            img_path = config.eval_dir / f"{name}.png"
        else:
            img_path = config.eval_dir / f"{name}.jpg"


        img = cv.imread(str(img_path))
        label_path = config.eval_dir / f"{name}.txt"
        eval_path = config.eval_dir / f"{name}.md"

        eval_item = pipeline.predict(img_path, iou_thresh=0.25, conf_thresh=0.25)
        eval_item = pipeline.topology(eval_item, threshold=0.25)

        gt = GroundTruth(eval_item.gt_bboxes, eval_path)
        pred = Prediction(eval_item.pred_bboxes, eval_item.topology)

        gt_adj, pred_adj = gt.adjacency, pred.adjacency
        evaluator = Evaluator(gt_adj, pred_adj)
        res = evaluator.evaluate()
        if res.FNS:
            macro_res.FNS += 1
        else:
            macro_res.TPS += 1

        overall_res += res

        p(res)

    p("Result:\n", overall_res)
    precision = overall_res.TPS / (overall_res.TPS + overall_res.FPS)
    recall = overall_res.TPS / (overall_res.TPS + overall_res.FNS)
    f1 = 2 * precision * recall / (precision + recall)

    p("Precision:", precision, "\nRecall:", recall, "\nF1:", f1)
    p("Macro recall:", macro_res.TPS / (macro_res.TPS + macro_res.FNS))


if __name__ == "__main__":
    main()
