from config import config
import utils
from augment import YoloAugmentator
import itertools as it


def main():
    # gts = YoloAugmentator.fileloader(config.test_out_dir)
    # gts = YoloAugmentator.fileloader(config.valid_out_dir)
    gts = YoloAugmentator.fileloader(config.test_out_dir)


    max_iou = 0
    for _, label in gts:
        labels = utils.load_ground_truth(label)
        bboxes = [utils.YoloBBox((608, 608)).from_ground_truth(l) for l in labels]

        for b1, b2 in it.combinations(bboxes, 2):
            iou = utils.calc_iou(b1.abs, b2.abs)

            if iou > max_iou:
                max_iou = iou

    print(max_iou)





if __name__ == "__main__":
    main()
