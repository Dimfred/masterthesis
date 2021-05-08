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

def make_mean_std_pretty(pretty, params):
    pretty_row = [""]
    for i in range(len(params)):
        pretty_row.append("mean")
        pretty_row.append("std")
    pretty += [pretty_row]

    return pretty

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
            f"experiements_yolo/offline_aug/offaug_P{int(P)}_F{int(F)}_R{int(R)}/run{run}/results_raw.txt"
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


def main():
    #####################
    #### lr_init ########
    #####################
    process_lr_init()

    #####################
    #### offline_aug ####
    #####################
    # process_offline_aug()

    #####################
    #### online_aug #####
    #####################
    # process_online_aug()


if __name__ == "__main__":
    main()
