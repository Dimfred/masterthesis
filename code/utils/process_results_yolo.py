import numpy as np

from tabulate import tabulate




def parse_results(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        # remove "-----" and header
        lines = lines[2:-1]
        lines = [line.split(" ") for line in lines]
        lines = [[item for item in line if item] for line in lines]

    mAPs = [float(mAP) for mAP in lines[0]]

    return mAPs


def process_lr_init():
    def build_paths(experiment_name, experiment_subname, experiment_param, runs):
        return [
            f"{experiment_name}/{experiment_subname}_{experiment_param}/run{run}/results_raw.txt"
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

    pretty = [["LR", "mAP@0.5_mean", "mAP@0.5_std", "mAP@0.75_mean", "mAP@0.75_std"]]
    for lr, runs in lr_runs:
        paths = build_paths("experiments_yolo/lr_init", "LR", lr, runs)
        lr_mAPs = np.vstack([parse_results(path) for path in paths])
        lr_mean = np.mean(lr_mAPs, axis=0)
        lr_std = np.std(lr_mAPs, axis=0)

        vals = [lr_mean[0], lr_std[0], lr_mean[1], lr_std[1]]
        vals = [f"{100 * val:.3f}" for val in vals]
        pretty += [[lr, *vals]]

    print(tabulate(pretty))


def main():
    #################
    #### lr_init ####
    #################
    process_lr_init()

    ##########################
    #### offline_aug_grid ####
    ##########################
    # process_lr_init()

if __name__ == "__main__":
    main()
