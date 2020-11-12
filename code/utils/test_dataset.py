import tensorflow as tf

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2 as cv
import numpy as np


def test_dataset(yolo, dataset):
    for i, (images, gt) in enumerate(dataset):
        for j in range(len(images)):
            _candidates = []
            for candidate in gt:
                grid_size = candidate.shape[1:3]
                _candidates.append(
                    tf.reshape(
                        candidate[j], shape=(1, grid_size[0] * grid_size[1] * 3, -1)
                    )
                )
            candidates = np.concatenate(_candidates, axis=1)

            frame = images[j, ...] * 255
            frame = frame.astype(np.uint8)

            pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0])
            pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            image = yolo.draw_bboxes(frame, pred_bboxes)
            cv.namedWindow("result")
            cv.resizeWindow("result", 1200, 1200)
            cv.imshow("result", image)
            while cv.waitKey(10) & 0xFF != ord("q"):
                pass
        if i == 10:
            break

    cv.destroyWindow("result")
