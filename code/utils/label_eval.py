import cv2 as cv
import shutil as sh
from pathlib import Path

from config import config
import utils


def create_imgs_with_idx(ds):
    for img_path, label_path in ds:
        img = cv.imread(str(img_path))
        labels = utils.Yolo.parse_labels(label_path)

        new_img_path = config.eval_dir / f"{img_path.stem}_eval{img_path.suffix}"
        try:
            sh.copy(img_path, new_img_path)
        except Exception as e:
            print(e)

        new_label_path = config.eval_dir / f"{label_path.stem}_eval{label_path.suffix}"
        try:
            sh.copy(label_path, new_label_path)
        except Exception as e:
            print(e)

        for idx, l in enumerate(labels):
            bbox = utils.YoloBBox(img.shape).from_ground_truth(l)

            x1, y1, x2, y2 = bbox.abs
            cv.putText(
                img, str(idx), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )

        new_img_path = config.eval_dir / f"{img_path.stem}_eval_idxs{img_path.suffix}"
        cv.imwrite(str(new_img_path), img)

        new_label_path = config.eval_dir / f"{label_path.stem}_eval{label_path.suffix}"
        try:
            sh.copy(label_path, new_label_path)
        except Exception as e:
            print(e)


def gt_topology_path_from_img(eval_img_path):
    gt_path = f"{eval_img_path.parent / eval_img_path.stem}.md"
    return Path(gt_path)


def write_gt_file(eval_img_path, adjacency, topology_str):
    n_components = len(adjacency[0]) // 2

    legend = []
    for i in range(n_components):
        legend.extend((i, i))
    legend = " ".join(str(i) for i in legend)

    # mat to space seperated string
    adjacency = [" ".join(str(v) for v in row) for row in adjacency]
    # add comment
    adjacency = [
        f"{row} // {topology_comment}"
        for row, topology_comment in zip(adjacency, topology_str)
    ]
    adjacency = "\n".join(adjacency)

    eval_img_with_idx_path = Path(
        f"{eval_img_path.parent / eval_img_path.stem}_idxs{eval_img_path.suffix}"
    )

    gt_topology_path = gt_topology_path_from_img(eval_img_path)
    with open(gt_topology_path, "w") as f:
        f.write(
            f"""
# {eval_img_path.name}
![img]({eval_img_with_idx_path.name})

always \\<left right> or \\<top bottom>

## START

\t{legend}
\t{adjacency}

## END

            """
        )


def main():
    # valid_ds = utils.Yolo.load_dataset(config.valid_out_dir)
    # create_imgs_with_idx(valid_ds)

    eval_ds = utils.Yolo.load_dataset(config.eval_dir)
    for eval_img_path, eval_label_path in eval_ds:
        print("Creating for:", eval_img_path.name)

        gt_topology_path = gt_topology_path_from_img(eval_img_path)
        # we have it already labeled
        if gt_topology_path.exists():
            continue

        row_len = len(utils.Yolo.parse_labels(eval_label_path)) * 2
        rows = []
        inputs = []

        inp = 0
        while True:
            inp = input("Input Topology: ")
            inp = inp.strip()
            if inp == "q":
                break
            inputs.append(inp)

            in_topology = inp.strip().split(",")
            in_topology = [s.strip() for s in in_topology]

            orientations = [s[-1] for s in in_topology]
            idxs = [s[:-1] for s in in_topology]

            row = [0 for _ in range(row_len)]

            for idx, orientation in zip(idxs, orientations):
                topology_idx = int(idx) * 2
                if orientation == "r" or orientation == "b":
                    topology_idx += 1

                row[topology_idx] = 1

            rows.append(row)

        adjacency = rows
        write_gt_file(eval_img_path, adjacency, inputs)


if __name__ == "__main__":
    main()
