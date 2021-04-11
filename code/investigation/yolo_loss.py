def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size t = stride *conv_shape(1)
    Conv_raw_prob = conv[..., 5:]
    Conf = [pred_xywh = pred, :, 0: :, :, :, :, 4:5]
    label_xywh = label[..., 0:4]
    response_bbox = label[..., 4:5]
    label_prob = label[..., 5:]
    ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), ​​axis=-1)
# (8, 13, 13, 3, 1)
    input_size = tf.cast(input_size, tf.float32 ) # The weight of each prediction box
    xxxiou_loss = 2-(ground truth area/picture area)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (input_size ** 2)
    ciou_loss = response_bbox * bbox_loss_scale * (1-ciou)
# 1. respond_bbox is used as a mask, and xxxiou_loss is calculated when there is an object
