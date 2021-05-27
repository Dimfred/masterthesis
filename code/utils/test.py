import utils




img_size = (500, 500)

def make_rel(bbox, img_size):
    h, w = img_size

    x1, y1, x2, y2 = bbox

    x1_rel, x2_rel = x1 / w, x2 / w
    y1_rel, y2_rel = y1 / h, y2 / h

    return (x1_rel, y1_rel, x2_rel, y2_rel)


bbox1 = (100, 100, 200, 200)
bbox2 = (150, 150, 300, 300)

iou1 = utils.nb_calc_iou(bbox1, bbox2)
iou2 = utils.nb_calc_iou(make_rel(bbox1, img_size), make_rel(bbox2, img_size))

print(iou1, iou2)
