# Experiments

!!!!!!!!!Experiment with data (reduce train set see how it behaves 1 0.8 0.6 0.4)

## General

    ~3 hours converging

    CLAHE := Contrast Limited Adaptive Histogram Equalization
    CLAHE(tile_grid_size, clip_limit)
        tile_grid_size := size of grid for histogram equalization
        clip_limit := upper threshold for contrast limiting
    Just want to use the default values for CLAHE and turn on off


## Definitions

    Plain Dataset = NO rotation NO flipping NO projection on checkered paper
    Rot Dataset = rotated
    Flip Dataset = flipped
    Proj Dataset = projection on checkered
    RotFlip Dataset = rotated and flipped
    Dataset = rotated, flipped, projections

## Initial LR Experiment

Performed on plain train / valid
BatchSize 64

LR      | Result (Best mAP)
0.50000 | DIV after 1k
0.25000 | DIV after 1k
0.10000 | DIV after 1k
0.05000 | DIV after 1k
0.02500 | rtx b0
0.01000 | rtx b1
0.00500 | rtx b2
0.00250 |
0.00100 |
0.00050 |
0.00025 |
0.00010 |


    LR        BEST ABOVE
    BatchSize 64

| Grid Dataset | V1    | V2     | NumParams |
| ------------ | --    | --     | --        |
| Proj         | True  | False  | 2         |
| Flip         | True  | False  | 2         |
| Rot          | True  | False  | 2         |
| ------------ | --    | --     | --        |
| N            | --    | --     | 8         |


Using best of above

| Ablation 2                         |       |        |     |           | NumParams
| ------------                       | --    | --     | --  | --        | --
| Rotate                             | 0°    | 10°    | 20° | 30°       | 4
| RandomScale                        | 0.0   | 0.2    | 0.4 | 0.6       | 4
| Rotate and Scale (best above)      | True  | False  | 2   |           |
| BBoxSafeCrop                       | True  | False  | 2   |           |
| CLAHE & ColorJitter                | True  | False  | 2   |           |

Using best of above

| Settings Grid Aug                  | V1    | V2     | NumParams     |
| ------------                       | --    | --     | --            |
| Batch                              | 32    | 64     | 2             |
| Loss                               | CIoU  | EIoU   | 2             |
| LR                                 |       |        | N             |
| ----------------------             | ----  | -----  | --            |
| Takes                              |       |        | 32 * N * 4h   |


lr = 3: => 12d
lr = 4: => 16d
lr = 5: => 20d
lr = 6: => 24d
lr = 7: => 28d
