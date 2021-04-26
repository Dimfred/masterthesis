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

## InitialInitial LR Test

Performed on plain train / valid
Loss      CIoU
BatchSize 64
Capped at 4000 steps

BurnIn     0 batches ALL DIV

BurnIn    10 batches (20 epochs)

LR      | Result (Best mAP)
0.01000 | DIV / Step 1994 / Loss 311
0.00500 | DIV / Step 1129 / Loss 200
0.00250 | DIV / Step 1121 / Loss 200
0.00100 | DIV / Step 1100 / Loss 200
0.00050 | DIV / Step 1310 / Loss 360
0.00025 | DIV / Step 1280 / Loss 370
0.00010 |

BurnIn    10 batches (20 epochs)

LR      | Result (Best mAP)
0.01000 | DIV / Step 1994 / Loss 311
0.00500 | DIV / Step 1129 / Loss 200
0.00250 | DIV / Step 1121 / Loss 200
0.00100 | DIV / Step 1100 / Loss 200
0.00050 | DIV / Step 1310 / Loss 360
0.00025 | DIV / Step 1280 / Loss 370
0.00010 | DIV / Step 2500 / Loss 315


BurnIn    500 batches (1000 epochs)

| LR      | Result (Best mAP) |
| -----   | ----------------- |
| 0.01000 | 82.28             |
| 0.00500 | 81.76             |
| 0.00250 | 81.18             |
| 0.00100 | 81.20             |
| 0.00050 | 79.36             |
| 0.00025 | 84.60             |
| 0.00010 | 82.60             |

BurnIn    1000 batches (2000 epochs)

| LR      | Result (Best mAP) |
| ------- | ----------------- |
| 0.01000 | 82.48             |
| 0.00500 | 82.24             |
| 0.00250 | 83.78             |
| 0.00100 | 81.20             |
| 0.00050 | 81.86             |
| 0.00025 | 79.83             |
| 0.00010 | 82.87             |


## Initial LR Test

    BurnIn     1000
    Batch      64
    MaxBatches 4000

| LR = 0.01    |
| ---------    |
| DIV          |

| LR = 0.005   |
| ----------   |
| DIV          |

| LR = 0.0025  | 1      | 2      | 3      |
| -----------  | ------ | ------ | ------ |
| diode_left   |
| diode_top    |
| diode_right  |
| diode_bot    |
| res_de_hor   |
| res_de_ver   |
| cap_hor      |
| cap_ver      |
| gr_left      |
| gr_top       |
| gr_right     |
| gr_bot       |
| ind_de_hor   |
| ind_de_ver   |
| source_hor   |
| source_ver   |
| current_hor  |
| current_ver  |
| text         |
| arrow_left   |
| arrow_top    |
| arrow_right  |
| arrow_bot    |
| OVERALL      |

| LR = 0.001   | 1      | 2      | 3      |
| ----------   | ------ | ------ | ------ |
| diode_left   |
| diode_top    |
| diode_right  |
| diode_bot    |
| res_de_hor   |
| res_de_ver   |
| cap_hor      |
| cap_ver      |
| gr_left      |
| gr_top       |
| gr_right     |
| gr_bot       |
| ind_de_hor   |
| ind_de_ver   |
| source_hor   |
| source_ver   |
| current_hor  |
| current_ver  |
| text         |
| arrow_left   |
| arrow_top    |
| arrow_right  |
| arrow_bot    |
| OVERALL      |

| LR = 0.0005  | 1      | 2      | 3      |
| -----------  | ------ | ------ | ------ |
| diode_left   |
| diode_top    |
| diode_right  |
| diode_bot    |
| res_de_hor   |
| res_de_ver   |
| cap_hor      |
| cap_ver      |
| gr_left      |
| gr_top       |
| gr_right     |
| gr_bot       |
| ind_de_hor   |
| ind_de_ver   |
| source_hor   |
| source_ver   |
| current_hor  |
| current_ver  |
| text         |
| arrow_left   |
| arrow_top    |
| arrow_right  |
| arrow_bot    |
| OVERALL      |

| LR = 0.00025 | 1      | 2      | 3      |
| ------------ | ------ | ------ | ------ |
| diode_left   |
| diode_top    |
| diode_right  |
| diode_bot    |
| res_de_hor   |
| res_de_ver   |
| cap_hor      |
| cap_ver      |
| gr_left      |
| gr_top       |
| gr_right     |
| gr_bot       |
| ind_de_hor   |
| ind_de_ver   |
| source_hor   |
| source_ver   |
| current_hor  |
| current_ver  |
| text         |
| arrow_left   |
| arrow_top    |
| arrow_right  |
| arrow_bot    |
| OVERALL      |

| LR = 0.0001  | 1      | 2      | 3      |
| -----------  | ------ | ------ | ------ |
| diode_left   |
| diode_top    |
| diode_right  |
| diode_bot    |
| res_de_hor   |
| res_de_ver   |
| cap_hor      |
| cap_ver      |
| gr_left      |
| gr_top       |
| gr_right     |
| gr_bot       |
| ind_de_hor   |
| ind_de_ver   |
| source_hor   |
| source_ver   |
| current_hor  |
| current_ver  |
| text         |
| arrow_left   |
| arrow_top    |
| arrow_right  |
| arrow_bot    |
| OVERALL      |


------------------------------------------------------------------------------

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
| Rotate                             | 10°   | 20°    | 30° |           | 4
| RandomScale                        | 0.2   | 0.4    | 0.6 |           | 4
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
