### darknet

    ./darknet detector train yolov4.data yolov4-tiny-custom.cfg checkpoints/xxx -map -show | tee log.txt
    tail --follow log.txt

    module load python3

    // cip
    ./darknet_cip detector train yolocip.data small_edge.cfg checkpoints_cip/last.weights -map -dont_show | tee log.txt

### calc anchors

    ./darknet calc_anchors .data -num_of_clusters 9 -width 608 -height 608

### map

    ./darknet detector map .data .cfg .weights -iou_threshold

### build

    yayi gcc8
    yayi opencv3-opt
