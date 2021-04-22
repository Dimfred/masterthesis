# Experiments

## General

    ~3 hours converging

    CLAHE := Contrast Limited Adaptive Histogram Equalization
    CLAHE(tile_grid_size, clip_limit)
        tile_grid_size := size of grid for histogram equalization
        clip_limit := upper threshold for contrast limiting
    Just want to use the default values for CLAHE and turn on off

## Settings

| Ablation               | V1    | V2     | NumParams |
| Grid Paper Projection  | True  | False  | 2         |
| Offline Rotate Flip    | True  | False  | 2         |


Using best of above

| Ablation 2             |       |        |     |           | NumParams
| ------------           | --    | --     | --  | --        |
| Rotate                 | 0°    | 10°    | 20° | 30°       | 4
| RandomScale            | 0.0   | 0.2    | 0.4 | 0.6       | 4


Using best of above

| Settings Grid Aug                  | V1    | V2     | NumParams     |
| ------------                       | --    | --     | --            |
| Rotate and Scale (best above)      | True  | False  | 2             |
| BBoxSafeCrop                       | True  | False  | 2             |
| CLAHE & ColorJitter                | True  | False  | 2             |
| Batch                              | 32    | 64     | 2             |
| Loss                               | CIoU  | EIoU   | 2             |
| LR                                 |       |        | N             |
| ----------------------             | ----  | -----  | --            |
| Takes                              |       |        | 32 * N * 3h   |


lr = 3: => 12d
lr = 4: => 16d
lr = 5: => 20d
lr = 6: => 24d
lr = 7: => 28d
