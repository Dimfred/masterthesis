# Experiments


## General

    ~3 hours converging

    CLAHE := Contrast Limited Adaptive Histogram Equalization
    CLAHE(tile_grid_size, clip_limit)
        tile_grid_size := size of grid for histogram equalization
        clip_limit := upper threshold for contrast limiting
    Just want to use the default values for CLAHE and turn on off

    !!!!!!!!!Experiment with data (reduce train set see how it behaves 1 0.8 0.6 0.4)

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

    mAP@50:75
    BurnIn    1000 batches (2000 epochs)

O okay
C cancelled
X NaN

| LR      | 0 | 1 | 2 |
| ------- | - | - | - |
| 0.01000 | O | O | O |
| 0.00500 | O | O | O |
| 0.00250 | O | O | O |
| 0.00100 | O | O | O |
| 0.00050 | O | O | O |
| 0.00025 | O | O | O |
| 0.00010 | O | O | O |

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

    LR        0.001
    BatchSize 64

| Proj | Flip | Rot | 0 | 1 | 2 |
| ---- | ---- | --- | - | - | - |
| 0    | 0    | 1   | O | O | O |
| 0    | 1    | 0   | O | O | O |
| 0    | 1    | 1   |   |   |   |
| 1    | 0    | 0   |   |   |   |
| 1    | 0    | 1   |   |   |   |
| 1    | 1    | 0   |   |   |   |
| 1    | 1    | 1   |   |   |   | WAS BEST!

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

Using best of above

| Ablation 2                  |       |        |     |     |     |NumParams
| --------------------------- | --    | --     | --  | --  | --  |--
| Rotate                      | 10°   | 20°    | 30° |     |     | 3
| RandomScale                 | 0.1   | 0.2    | 0.3 |     |     | 3
| BBoxSafeCrop                | 0.9   | 0.8    | 0.7 |     |     | 1
| ColorJitter                 | 0.1   | 0.2    | 0.3 |     |     | 3
|                             |       |        |     |     |     |
| N                           |       |        |     |     |     | 15

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

| Settings Grid Aug  | V1    | V2      | NumParams     |
| ------------------ | --    | --      | --            |
| Batch              | 32    | 64      | 2             |
| Loss               | CIoU  | EIoU    | 2             |
| Activation         | Leaky | HSwiswh |
| LR                 |       |         | N             |
| ------------------ | ----  | -----   | --            |
| Takes              |       |         | 32 * N * 4h   |

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

| bs | loss    | lr        |
| -- | ------- | --------- |
| 32 | ciou    | 0.01      |
| 32 | ciou    | 0.005     |
| 32 | ciou    | 0.0025    |
| 32 | ciou    | 0.001     |
| 32 | ciou    | 0.0005    |
| 32 | ciou    | 0.00025   |
| 32 | ciou    | 0.0001    |
| 32 | eiou1   | 0.01      |
| 32 | eiou1   | 0.005     |
| 32 | eiou1   | 0.0025    |
| 32 | eiou1   | 0.001     |
| 32 | eiou1   | 0.0005    |
| 32 | eiou1   | 0.00025   |
| 32 | eiou1   | 0.0001    |
| 32 | eiou0.5 | 0.01      |
| 32 | eiou0.5 | 0.005     |
| 32 | eiou0.5 | 0.0025    |
| 32 | eiou0.5 | 0.001     |
| 32 | eiou0.5 | 0.0005    |
| 32 | eiou0.5 | 0.00025   |
| 32 | eiou0.5 | 0.0001    |
| 64 | ciou    | 0.01      |
| 64 | ciou    | 0.005     |
| 64 | ciou    | 0.0025    |
| 64 | ciou    | 0.001     |
| 64 | ciou    | 0.0005    |
| 64 | ciou    | 0.00025   |
| 64 | ciou    | 0.0001    |
| 64 | eiou1   | 0.01      |
| 64 | eiou1   | 0.005     |
| 64 | eiou1   | 0.0025    |
| 64 | eiou1   | 0.001     |
| 64 | eiou1   | 0.0005    |
| 64 | eiou1   | 0.00025   |
| 64 | eiou1   | 0.0001    |
| 64 | eiou0.5 | 0.01      |
| 64 | eiou0.5 | 0.005     |
| 64 | eiou0.5 | 0.0025    |
| 64 | eiou0.5 | 0.001     |
| 64 | eiou0.5 | 0.0005    |
| 64 | eiou0.5 | 0.00025   |
| 64 | eiou0.5 | 0.0001    |

O p3 train.py 64 ciou 0.01 0;       p3 train.py 64 ciou 0.01 1;       p3 train.py 64 ciou 0.01 2;
O p3 train.py 64 ciou 0.005 0;      p3 train.py 64 ciou 0.005 1;      p3 train.py 64 ciou 0.005 2;
O p3 train.py 64 ciou 0.0025 0;     p3 train.py 64 ciou 0.0025 1;     p3 train.py 64 ciou 0.0025 2;
O p3 train.py 64 ciou 0.001 0;      p3 train.py 64 ciou 0.001 1;      p3 train.py 64 ciou 0.001 2;
O p3 train.py 64 ciou 0.0005 0;     p3 train.py 64 ciou 0.0005 1;     p3 train.py 64 ciou 0.0005 2;
O p3 train.py 64 ciou 0.00025 0;    p3 train.py 64 ciou 0.00025 1;    p3 train.py 64 ciou 0.00025 2;
O p3 train.py 64 ciou 0.0001 0;     p3 train.py 64 ciou 0.0001 1;     p3 train.py 64 ciou 0.0001 2;

O p3 train.py 64 eiou1 0.01 0;      p3 train.py 64 eiou1 0.01 1;      p3 train.py 64 eiou1 0.01 2;
O p3 train.py 64 eiou1 0.005 0;     p3 train.py 64 eiou1 0.005 1;     p3 train.py 64 eiou1 0.005 2;
O p3 train.py 64 eiou1 0.0025 0;    p3 train.py 64 eiou1 0.0025 1;    p3 train.py 64 eiou1 0.0025 2;
O p3 train.py 64 eiou1 0.001 0;     p3 train.py 64 eiou1 0.001 1;     p3 train.py 64 eiou1 0.001 2;
O p3 train.py 64 eiou1 0.0005 0;    p3 train.py 64 eiou1 0.0005 1;    p3 train.py 64 eiou1 0.0005 2;
O p3 train.py 64 eiou1 0.00025 0;   p3 train.py 64 eiou1 0.00025 1;   p3 train.py 64 eiou1 0.00025 2;
O p3 train.py 64 eiou1 0.0001 0;    p3 train.py 64 eiou1 0.0001 1;    p3 train.py 64 eiou1 0.0001 2;

O p3 train.py 64 eiou0.5 0.01 0;    p3 train.py 64 eiou0.5 0.01 1;    p3 train.py 64 eiou0.5 0.01 2;
O p3 train.py 64 eiou0.5 0.005 0;   p3 train.py 64 eiou0.5 0.005 1;   p3 train.py 64 eiou0.5 0.005 2;
O p3 train.py 64 eiou0.5 0.0025 0;  p3 train.py 64 eiou0.5 0.0025 1;  p3 train.py 64 eiou0.5 0.0025 2;
O p3 train.py 64 eiou0.5 0.001 0;   p3 train.py 64 eiou0.5 0.001 1;   p3 train.py 64 eiou0.5 0.001 2;
O p3 train.py 64 eiou0.5 0.0005 0;  p3 train.py 64 eiou0.5 0.0005 1;  p3 train.py 64 eiou0.5 0.0005 2;
O p3 train.py 64 eiou0.5 0.00025 0; p3 train.py 64 eiou0.5 0.00025 1; p3 train.py 64 eiou0.5 0.00025 2;
O p3 train.py 64 eiou0.5 0.0001 0;  p3 train.py 64 eiou0.5 0.0001 1;  p3 train.py 64 eiou0.5 0.0001 2;

----------------------------------------------------------------------

## NMS and TTA Experiments

WBF := Weighted Bounding Box Fusion

|         | Default                           | WBF                                    | TTA + DIoU           | TTA + WBF            |
| ------- | --------------------------------- | -------------------------------------- | -------------------- | -------------------- |
|         | raw = yolo(input)                 | raw = yolo(input)                      | inputs = TTA(input)  | inputs = TTA(input)  |
|         | pred = DIoU_NMS(raw)              | pred = WBF(raw)                        | raws = yolo(inputs)  | raws = yolo(inputs)  |
|         |                                   |                                        | pred = DIoU(raws)    | pred = WBF(raws)     |
| Comment | Bit worse than wbf                | So far the best                        | Super bad            | Still bad            |

|         | TTA + WBF + Voting                | TTA + DIoU + WBF + Voting              |                      |                      |
| ------- | --------------------------------- | -------------------------------------- | -------------------- | -------------------- |
|         | inputs = TTA(input)               | inputs = TTA(input)                    |                      |                      |
|         | raws = yolo(inputs)               | raws = yolo(inputs)                    |                      |                      |
|         | preds = WBF_Votes(raw, min_votes) | post_raws = DIoU(raws)                 |                      |                      |
|         |                                   | pred = WBF_Votes(post_raws, min_votes) |                      |                      |
| Comment |                                   |                                        |                      |                      |

SUPER WEIRD:

| type     | result                      |
| ----     | --------------------------- |
| padded   | mAP(diou / wbf) >> mAP(tta) |
| original | mAP(tta) >> mAP(diou / wbf) |



img = h : w

padded = s : s


// not tta

res = yolo(padded)
mAP() => guter map

fitted_res = fit(res) // remove padding
mAP() => scheise


w = 800

x = 100


x_rel = x / w = 0.1
