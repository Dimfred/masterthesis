from utils import Yolo
import cv2 as cv
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from config import config
import utils
from utils import YoloBBox

import time
import random


class MNistLoader:
    def __init__(self):
        self.data = {str(n): [] for n in range(10)}

    @utils.stopwatch("Mnist::load")
    def load(self, mnist_dir, worker=16, roi_only=False):
        # start_load = time.perf_counter()

        def load_and_store(number, path):
            img = self._imread(path)
            img = 255 - img
            self.data[str(number)].append(img)

        with ThreadPoolExecutor(max_workers=worker) as executor:
            for number in range(10):
                number_dir = mnist_dir / str(number)
                pngs = number_dir.glob("**/*.png")

                for png in pngs:
                    executor.submit(load_and_store, number, png)

        end_load = time.perf_counter()
        # print("MNist loading took:", end_load - start_load)

        if roi_only:
            self.extract_rois()

    def extract_rois(self):
        data = self.data
        for number in range(10):
            number_imgs = data[str(number)]
            for idx, img in enumerate(number_imgs):
                grey_idxs = np.argwhere(img < 255)
                tl, br = self._bbox_from_number(grey_idxs)
                (y1, x1), (y2, x2) = tl, br

                roi = img[y1:y2, x1:x2]

                # replace original with roi only
                data[str(number)][idx] = roi

                # DEBUG
                # utils.show(img, roi)

    def _bbox_from_number(self, grey_idxs):
        ys, xs = grey_idxs[:, 0], grey_idxs[:, 1]

        top, bot = np.min(ys), np.max(ys)
        left, right = np.min(xs), np.max(xs)

        return (top, left), (bot, right)

    def _imread(self, path):
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        return img


class Annotation:
    def __init__(self, start, img, bboxes):
        self.start = start
        self.img = img
        self.bboxes = bboxes


class Annotator:
    def __init__(self, dataset, name2idx, number_limits=(1, 999)):
        self.dataset = dataset
        self.name2idx = name2idx

        # defined in sub class
        self.unit = None
        self.sub_units = ["", "n", "m", "k", "M", "G"]

        self.number_limits = number_limits

        self.h_min = 10
        self.w_min = None  # TODO?

    def _random_annotation_str(self):
        number = random.randint(*self.number_limits)
        sub_unit = random.choice(self.sub_units)

        return str(number)
        # TODO
        # return f"{number}{sub_unit}{self.unit}"

    def _convert_to_imgs(self, annotation_str):
        annotation_imgs = []
        for c in annotation_str:
            # c_dataset = self.dataset[c]
            c_dataset = self.dataset[c]
            c_img = random.choice(c_dataset)
            annotation_imgs.append(c_img)

        return annotation_imgs

    def generate(self, img, cls_orientation, bbox_to_annotate):
        annotation_str = self._random_annotation_str()
        annotation_imgs = self._convert_to_imgs(annotation_str)

        return annotation_imgs

    def make_random_annotation(self, bbox, cls_orientation):
        # if one either h or w is None the aspect ratio of each annotation img is kept
        h_annotation, w_annotation = self.calculate_annotation_size(
            bbox, cls_orientation
        )

        if h_annotation < self.h_min:
            return None

        # generate random annotation in the format <number><sub_unit><unit>
        # e.g. for resister 100kOhm
        annotation_str = self._random_annotation_str()
        # select a random img for each char in the annotation sequence
        annotation_imgs = self._convert_to_imgs(annotation_str)
        # TODO is the original image changed on resize? probably not
        annotation_imgs = [
            utils.resize(img, width=w_annotation, height=h_annotation)
            for img in annotation_imgs
        ]

        annotation_img = np.concatenate(annotation_imgs, axis=1)
        w_annotation = annotation_img.shape[1]

        annotation_start = self.calculate_start_coordinates(
            bbox, h_annotation, w_annotation, cls_orientation
        )
        if annotation_start is None:
            return None

        annotation_bboxes = self.generate_bboxes(
            annotation_start, annotation_imgs, annotation_str, bbox.img_dim
        )

        annotation = Annotation(
            start=annotation_start, img=annotation_img, bboxes=annotation_bboxes
        )
        return annotation

    def generate_bboxes(
        self, annotation_start, annotation_imgs, annotation_str, img_dim
    ):
        offset = 0

        y_start, x_start = annotation_start

        bboxes = []
        x_cur = 0
        for char, img in zip(annotation_str, annotation_imgs):
            ih, iw = img.shape

            x1 = x_start + x_cur - offset
            y1 = y_start - offset
            x2 = x1 + iw + 2 * offset
            y2 = y1 + ih + 2 * offset

            label_idx = self.name2idx[char]

            x_cur += iw

            bbox = YoloBBox(img_dim).from_abs(x1, y1, x2, y2, label_idx)
            bboxes.append(bbox)

        return bboxes

    def calculate_annotation_size(self, bbox, orientation):
        w_bbox, h_bbox = bbox.abs_dim
        print("-----------------------------")
        print("AbsDim", bbox.abs_dim)
        print("RelDim", bbox.h, bbox.w)
        scaler = random.uniform(0.25, 0.4)
        print("Scaler", scaler)

        if orientation == "hor":
            h_annotation = int(scaler * h_bbox)
            w_annotation = None
        else:
            h_annotation = int(scaler * w_bbox)
            w_annotation = None
        print("AbsH", h_annotation)

        return h_annotation, w_annotation

    def calculate_start_coordinates(
        self, bbox, h_annotation, w_annotation, orientation
    ):
        ih, iw = bbox.img_dim
        mx_bbox, my_bbox = bbox.abs_mid
        x1, y1, x2, y2 = bbox.abs

        if orientation == "hor":
            x_annotation_start = int(mx_bbox) - (w_annotation // 2)
            if x_annotation_start < 0:
                return None

            # top
            if random.uniform(0, 1) <= 0.5:
                y_annotation_start = y1 - 2 - h_annotation
                if y_annotation_start < 0:
                    return None
            # bot
            else:
                y_annotation_start = y2 + 2
        else:
            y_annotation_start = int(my_bbox) - (h_annotation // 2)
            if y_annotation_start < 0:
                return None

            # left
            if random.uniform(0, 1) <= 0.5:
                x_annotation_start = x1 - 2 - w_annotation
                if x_annotation_start < 0:
                    return None
            # right
            else:
                x_annotation_start = x2 + 2

        # aditionally verify positive img bounds
        x_annotation_end = x_annotation_start + w_annotation
        if x_annotation_end > iw:
            return None

        y_annotation_end = y_annotation_start + h_annotation
        if y_annotation_end > ih:
            return None

        return y_annotation_start, x_annotation_start


class ResistorAnnotator(Annotator):
    def __init__(self, dataset, name2idx):
        super().__init__(dataset, name2idx)
        self.unit = "R"


class CapacitorAnnotator(Annotator):
    def __init__(self, dataset, name2idx):
        super().__init__(dataset, name2idx)
        self.unit = "F"


class InductorAnnotator(Annotator):
    def __init__(self, dataset, name2idx):
        super().__init__(dataset, name2idx)
        self.unit = "H"


class SourceAnnotator(Annotator):
    def __init__(self, dataset, name2idx):
        super().__init__(dataset, name2idx)
        self.unit = "V"


class CurrentAnnotator(Annotator):
    def __init__(self, dataset, name2idx):
        super().__init__(dataset, name2idx)
        self.unit = "A"


class AnnotatorFactory:
    def __init__(self, dataset, name2idx):
        self.dataset = dataset
        self.name2idx = name2idx

        self.generators = {
            "res": ResistorAnnotator(dataset, name2idx),
            "cap": CapacitorAnnotator(dataset, name2idx),
            "ind": InductorAnnotator(dataset, name2idx),
            "source": SourceAnnotator(dataset, name2idx),
            "current": CurrentAnnotator(dataset, name2idx),
        }

    def get(self, cls_type):
        return self.generators.get(cls_type, None)


class AnnotationProjector:
    def __init__(self, mnist_dataset, classes, p=1.0):
        self.dataset = mnist_dataset
        self.classes = classes
        self.name2idx = {idx: name for idx, name in enumerate(self.classes)}

        # add numbers
        n_classes = len(classes)
        for n in range(10):
            self.name2idx[str(n)] = n_classes
            n_classes += 1

        self.p = p

        self.annotator_factory = AnnotatorFactory(self.dataset, self.name2idx)

    @utils.stopwatch("Projector::project")
    def project(self, img, labels):
        bboxes = [YoloBBox(img.shape).from_ground_truth(label) for label in labels]

        for bbox_idx in range(len(bboxes)):
            bbox = bboxes[bbox_idx]
            if random.uniform(0, 1) > self.p:
                continue

            cls_name = self.classes[bbox.label]
            cls_type, *_, cls_orientation = cls_name.split("_")

            annotator = self.annotator_factory.get(cls_type)
            # no annotator exists for this cls
            if annotator is None:
                continue

            annotation = annotator.make_random_annotation(bbox, cls_orientation)
            if annotation is None:
                continue

            if not self.are_bboxes_valid(bboxes, annotation.bboxes):
                # print("REJECTED")
                # timg = self._project(img.copy(), annotation.img, annotation.start)
                # utils.show_bboxes(timg, annotation.bboxes, type_="utils")
                continue

            img = self._project(img, annotation.img, annotation.start)
            bboxes.extend(annotation.bboxes)

            utils.show_bboxes(img, annotation.bboxes, type_="utils")
            # utils.show(img)

    def _project(self, img, annotation, p_start):
        y_start, x_start = p_start
        where_annotation = np.argwhere(annotation < 180)

        for y_annotation, x_annotation in where_annotation:
            y_in_img = y_annotation + y_start
            x_in_img = x_annotation + x_start
            img[y_in_img, x_in_img] = annotation[y_annotation, x_annotation]

        return img

    def are_bboxes_valid(self, bboxes, annotated_bboxes):
        # TODO this is brute force :(
        for present_bbox in bboxes:
            for new_bbox in annotated_bboxes:
                iou = utils.calc_iou(present_bbox.abs, new_bbox.abs)
                if iou > 1e-5:
                    return False

        return True


if __name__ == "__main__":
    print("test")
    loader = MNistLoader()
    print("other")
    # loader.load(config.mnist_train_dir, roi_only=True)
    loader.load(config.mnist_test_dir, roi_only=True)

    classes = utils.Yolo.parse_classes(config.yolo.classes)
    # dataset = utils.Yolo.load_dataset(config.train_out_dir)
    dataset = utils.Yolo.load_dataset(config.valid_out_dir)

    projector = AnnotationProjector(loader.data, classes)

    for img_path, label_path in dataset:
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        rh, rw = img.shape
        if rh > rw:
            rh, rw = 608, None
        else:
            rw, rh = 608, None

        img = utils.resize(img, width=rw, height=rh)
        labels = utils.Yolo.parse_labels(label_path)

        projector.project(img, labels)
        # utils.show(img)

    for i in range(20):
        utils.show(projector.dataset[str(0)][i])
