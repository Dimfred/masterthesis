import cv2 as cv

from config import config
import utils


def timeit(f):
    import time
    def deco(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print(end - start)
        return res

    return deco

@timeit
def load(imgs):
    # ~6gig
    return [cv.imread(str(img)) for img in imgs]

@timeit
def copy(imgs):
    return [img.copy() for img in imgs]


imgs = utils.list_imgs(config.train_out_dir)
# ~16s
imgs = load(imgs)

# ~2s => 8x faster
imgs_cpy = copy(imgs)


while True: pass
