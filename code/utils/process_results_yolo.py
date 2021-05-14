import numpy as np

from tabulate import tabulate

import utils


def parse_results(paths):
    all_pred_50, all_pred_75 = [], []
    for path in paths:
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            # remove "-----" and header
            lines = lines[2:-1]
            lines = [line.split(" ") for line in lines]
            lines = [[item for item in line if item] for line in lines]

        cls_names = ["mAP"] + [row[0] for row in lines[1:]]

        mAP_vals = [float(val) for val in lines[0]]
        cls_vals = [np.array([float(val) for val in row[1:]]) for row in lines[1:]]
        pred_vals = np.vstack([mAP_vals, cls_vals])

        pred_50, pred_75 = pred_vals[:, 0], pred_vals[:, 1]
        all_pred_50.append(pred_50.reshape(-1, 1))
        all_pred_75.append(pred_75.reshape(-1, 1))

    return np.hstack(all_pred_50), np.hstack(all_pred_75), cls_names


def mean_and_std(res):
    mean = res.mean(axis=1)
    std = res.std(axis=1)

    return mean.reshape(-1, 1), std.reshape(-1, 1)


def ffloat(val):
    return f"{100* val:.3f}"

    # return [f"{100 * val:.3f}" for val in iterable]
    # return [f"{val:.3f}" for val in iterable]


def make_red_green(pretty, cls_names, results):
    for cls_name, result_row in zip(cls_names, results):
        pretty_row = []
        for i, val in enumerate(result_row):
            if i % 2 == 0:
                pretty_row.append(utils.green(ffloat(val)))
            else:
                pretty_row.append(utils.red(ffloat(val)))

        pretty += [[cls_name, *pretty_row]]

    return pretty


def dump_csv(path, results):
    with open(path, "w") as f:
        results = [",".join(str(v) for v in res_) for res_ in results]
        for line in results:
            print(line, file=f)


def zip_with_names(cls_names, results):
    out = []
    for name, res in zip(cls_names, results):
        out += [[name, *res]]

    return out


def combine_50_75(res_50, res_75):
    combined = []
    for line_50, line_75 in zip(res_50, res_75):
        line = [[*line_50, "", "", "", *line_75]]
        combined += line

    return combined


def dump_lr_init():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/lr_init/LR_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = (
        # (0.01, (0, 1, 2)),
        (0.005, (0, 1, 2)),
        (0.0025, (0, 1, 2)),
        (0.001, (0, 1, 2)),
        (0.0005, (0, 1, 2)),
        (0.00025, (0, 1, 2)),
        (0.0001, (0, 1, 2)),
    )

    results = []
    for lr, runs in lr_runs:
        paths = build_paths(lr, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"LR@0.5: {lr}", "", "mean", "std", "",
                f"LR@0.75: {lr}", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/lr_init/results.csv", results)


def dump_offline_aug():
    def build_paths(P, F, R, runs):
        return [
            f"experiments_yolo/offline_aug/offaug_P{int(P)}_F{int(F)}_R{int(R)}/run{run}/results_raw.txt"
            for run in runs
        ]

    # (P, F, R), (runs)
    params_runs = (
        ((0, 0, 1), (0, 1, 2)),
        ((0, 1, 0), (0, 1, 2)),
        ((0, 1, 1), (0, 1, 2)),
        ((1, 0, 0), (0, 1, 2)),
        ((1, 0, 1), (0, 1, 2)),
        ((1, 1, 0), (0, 1, 2)),
        ((1, 1, 1), (0, 1, 2)),
    )

    results = []
    for (P, F, R), runs in params_runs:
        paths = build_paths(P, F, R, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"P{P}F{F}R{R}@0.5", "", "", "", "mean", "std", "",
                f"P{P}F{F}R{R}@0.75", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on

        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/offline_aug/results.csv", results)

def dump_offline_baseline_aug():
    def build_paths(runs):
        return [
            f"experiments_yolo/offline_baseline/offline_baseline/run{run}/results_raw.txt"
            for run in runs
        ]

    # (P, F, R), (runs)
    params_runs = (
        ((0, 1, 1), (0, 1, 2)),
        ((1, 1, 1), (0, 1, 2)),
    )

    results = []
    for runs in params_runs:
        paths = build_paths(runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"OffBase@0.5", "", "", "", "mean", "std", "",
                f"OffBase@0.5:0.75", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on

        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/offline_baseline/results.csv", results)

def dump_rotate_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/rotate/rotate_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = (
        (10, (0, 1, 2, 3, 4)),
        (20, (0, 1, 2, 3, 4)),
        (30, (0, 1, 2, 3, 4)),
        (40, (0, 1, 2, 3, 4)),
    )

    results = []
    for rot, runs in lr_runs:
        paths = build_paths(rot, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"Rotation@0.5: {rot}", "", "", "", "mean", "std", "",
                f"Roationt@0.5:0.75: {rot}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/rotate/results.csv", results)


def dump_random_scale_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/random_scale/random_scale_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = (
        (0.1, (0, 1, 2, 3, 4)),
        (0.2, (0, 1, 2, 3, 4)),
        (0.3, (0, 1, 2, 3, 4)),
        (0.4, (0, 1, 2, 3, 4)),
        (0.5, (0, 1, 2, 3, 4)),
    )

    results = []
    for scale, runs in lr_runs:
        paths = build_paths(scale, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"Scale@0.5: {scale}", "", "", "", "mean", "std", "",
                f"Scale@0.5:0.75: {scale}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/random_scale/results.csv", results)


def dump_color_jitter_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/color_jitter/color_jitter_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = (
        (0.1, (0, 1, 2, 3, 4)),
        (0.2, (0, 1, 2, 3, 4)),
        (0.3, (0, 1, 2, 3, 4)),
        (0.4, (0, 1, 2, 3, 4)),
        (0.5, (0, 1, 2, 3, 4)),
    )

    results = []
    for color_jitter, runs in lr_runs:
        paths = build_paths(color_jitter, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"ColorJitter@0.5: {color_jitter}", "", "", "", "mean", "std", "",
                f"ColorJitter@0.5:0.75: {color_jitter}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/color_jitter/results.csv", results)


def dump_bbox_safe_crop_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/bbox_safe_crop/bbox_safe_crop_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = ((1, (0, 1, 2, 3, 4)),)

    results = []
    for sc, runs in lr_runs:
        paths = build_paths(sc, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"SafeCrop@0.5: {sc}", "", "", "", "mean", "std", "",
                f"SafeCrop@0.5:0.75: {sc}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/bbox_safe_crop/results.csv", results)


def dump_clahe_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/clahe/clahe_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = ((1, (0, 1, 2, 3, 4)),)

    results = []
    for sc, runs in lr_runs:
        paths = build_paths(sc, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"Clahe@0.5: {sc}", "", "", "", "mean", "std", "",
                f"Clahe@0.5:0.75: {sc}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/clahe/results.csv", results)


def dump_gaussian_noise_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/gaussian_noise/gaussian_noise_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = ((1, (0, 1, 2, 3, 4)),)

    results = []
    for sc, runs in lr_runs:
        paths = build_paths(sc, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"GaussianNoise@0.5: {sc}", "", "", "", "mean", "std", "",
                f"GaussianNoise@0.5:0.75: {sc}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/gaussian_noise/results.csv", results)


def dump_blur_aug():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/blur/blur_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = ((3, (0, 1, 2, 3, 4)), (5, (0, 1, 2, 3, 4)))

    results = []
    for sc, runs in lr_runs:
        paths = build_paths(sc, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        # fmt: off
        results += [
            [],
            [
                f"Blur@0.5: {sc}", "", "", "", "mean", "std", "",
                f"Blur@0.5:0.75: {sc}", "", "", "", "mean", "std"],
            [],
        ]
        # fmt: on
        res_50 = zip_with_names(cls_names, pred_50)
        res_75 = zip_with_names(cls_names, pred_75)
        combined = combine_50_75(res_50, res_75)
        results += combined

    dump_csv("experiments_yolo/blur/results.csv", results)


def main():
    #####################
    #### lr_init ########
    #####################
    # process_lr_init()
    # dump_lr_init()

    #####################
    #### offline_aug ####
    #####################
    dump_offline_aug()
    dump_offline_baseline_aug()

    #####################
    #### online_aug #####
    #####################
    # dump_rotate_aug()
    # dump_random_scale_aug()
    # dump_color_jitter_aug()
    # dump_bbox_safe_crop_aug()
    # dump_clahe_aug()
    # dump_gaussian_noise_aug()
    # dump_blur_aug()


if __name__ == "__main__":
    main()


def process_lr_init():
    def build_paths(experiment_param, runs):
        return [
            f"experiments_yolo/lr_init/LR_{experiment_param}/run{run}/results_raw.txt"
            for run in runs
        ]

    lr_runs = (
        (0.01, (0, 1, 2, 3, 4)),
        (0.005, (0, 2, 3, 4, 5)),
        (0.0025, (0, 1, 2, 3, 4)),
        (0.001, (0, 1, 2, 3, 4)),
        (0.0005, (0, 1, 2, 3, 4)),
        (0.00025, (0, 1, 2, 3, 4)),
        (0.0001, (0, 1, 2, 3, 4)),
    )

    pretty = [""]
    for lr, _ in lr_runs:
        pretty.append(f"lr:{lr}")
        pretty.append("")
    pretty = [pretty]

    pretty = make_mean_std_pretty(pretty, lr_runs)

    # create results
    results = []
    for lr, runs in lr_runs:
        paths = build_paths(lr, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        mean, std = mean_and_std(pred_50)
        results += (mean, std)
    results = np.hstack(results)

    pretty = make_red_green(pretty, cls_names, results)
    print(tabulate(pretty))


def process_offline_aug():
    def build_paths(P, F, R, runs):
        return [
            f"experiments_yolo/offline_aug/offaug_P{int(P)}_F{int(F)}_R{int(R)}/run{run}/results_raw.txt"
            for run in runs
        ]

    # (P, F, R), (runs)
    params_runs = (
        ((0, 0, 1), (0, 1, 2, 3, 4)),
        ((0, 1, 0), (0, 1, 2, 3, 4)),
        ((0, 1, 1), (0, 1, 2, 3, 4)),
        ((1, 0, 0), (0, 1, 2, 3, 4)),
        ((1, 0, 1), (0, 1, 2, 3, 4)),
        ((1, 1, 0), (0, 1, 2, 3, 4)),
        ((1, 1, 1), (0, 1, 2, 3, 4)),
    )

    pretty = [""]
    for (P, F, R), _ in params_runs:
        pretty.append(f"P{P}F{F}R{R}")
        pretty.append("")
    pretty = [pretty]

    pretty = make_mean_std_pretty(pretty, params_runs)

    results = []
    for (P, F, R), runs in params_runs:
        paths = build_paths(P, F, R, runs)
        pred_50, pred_75, cls_names = parse_results(paths)

        mean, std = mean_and_std(pred_50)
        results += (mean, std)
    results = np.hstack(results)

    pretty = make_red_green(pretty, cls_names, results)
    print(tabulate(pretty))


def make_mean_std_pretty(pretty, params):
    pretty_row = [""]
    for i in range(len(params)):
        pretty_row.append("mean")
        pretty_row.append("std")
    pretty += [pretty_row]

    return pretty
