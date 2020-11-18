## Dataset

- show file_legend
- show train / valid

## Training

- in: 608x608x1
- 3 yolo layer (yolo small conf)
- batch_size: 21
- momentum: 0.9
- decay: 0.0005
- lr: 0.00026 (default 0.0026)
- wenn loss plateut lr / 10
- 1k epochs burn_in
- loss: ciou

augmentation

- scales +- 0.1
- angle 15 (vielleicht noch mehr drin)
- saturation 1.5
- exposure 1.5
- hue 0.1
- mosaic
- NO flip
- NO cutout

nicht ausgecheck

- anchor recalculation
- yolo layer (nms, jitter, scale_x_y, resize, beta_nms, greedynms)

## Trainierte Netze

- mit edges
- ohne edges
- classes stripped
- TODO classes tripped edges

## Performance

- show map calc
- show errors

## CV

- zeig alles
- Probleme vielleicht zu grosse Lücken (Obergrenze fuer dilations)
- Draht path fitting? Wie besser machen?
- Komplett auf CV scheissen, wenn T und Edges deluxe sind

## Probleme

- Draht fitting (Abbild scan / ltspice)
    - Ansaetze
    - clustern von linien
    - kmeans / hdbscan
    - finde Pfad zwischen Nodes
        - Nodes > 2 => fitte Pfad an bestehenden Pfad
        - *zeig Problem*

## LTSpice

- läuft.

## TODO

- Mehr Daten mit dickem Stift
- Metriken auslagern? Checken wie darknet es macht
- Avg class confidence
- Backgrounds
    1. Karo projecten
    1.

- OCR the shit out of the circuits
    1. ocr plain
    1. project letters on the dataset
    1. test: yolo with ocr directly? (model complexity)



## Literatur

- ANN (ann_geometric_features, nicht sehr gut)
    - Bauteil und Annotations extraktion
    - Einfach nur NN
    - Image Moments als features
    - 0.86 Precision
    - 1.0 Recall

- CV HOG SVM (cv_seg_hog_svm)
    - Erst segmentieren der Bauteile mit CV (e.g. region filling)
    - HOG features von der Segmentierung
    - len(HOG) wichtig bei der Wahl der accuracy
    - 87% auf 150 Imgs, 26 img / class

- Online recog with HMM
    - not really related?
    - no segmentation, class modeled through n consecutive strokes
        - könnte man bestimmt mit RNN / CRNN auch machen

-
