### darknet

    ./darknet detector train yolov4.data yolov4-tiny-custom.cfg checkpoints/xxx -map -show | tee log.txt
    tail --follow log.txt

    module load python3

    // cip
    ./darknet_cip detector train yolocip.data small_edge.cfg checkpoints_cip/last.weights -map -dont_show | tee log.txt

### build 

    yayi gcc8
    yayi opencv3-opt

