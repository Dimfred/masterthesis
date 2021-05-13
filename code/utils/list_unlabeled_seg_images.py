import utils
from config import config

def list_unlabeled(dir_):
    print("-----------------------------------------")
    print("-----------------------------------------")
    print(dir_)

    img_paths = utils.list_imgs(dir_)
    for img_path in img_paths:
        print(img_path.name)
    # for img_path in img_paths:
    #     seg_path = utils.segmentation_label_from_img(img_path)
    #     if not seg_path.exists():
    #         print(img_path.name)


list_unlabeled(config.train_dir)
list_unlabeled(config.valid_dir)
list_unlabeled(config.test_dir)
