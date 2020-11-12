### darknet

    ./darknet detector train yolov4.data yolov4-tiny-custom.cfg checkpoints/xxx -map -show | tee log.txt
    tail --follow log.txt

    module load python3

    // cip
    ./darknet_cip detector train yolov4.data small_conf.cfg weights/yolov4-tiny.conv.29 -map -dont_show | tee log.txt

### build 

    yayi gcc8
    yayi opencv3-opt

