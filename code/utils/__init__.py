import cv2 as cv

WINDOW_NAME = "img"


def show(img, width=600):
    cv.namedWindow(WINDOW_NAME)
    cv.moveWindow(WINDOW_NAME, 100, 100)
    img = resize(img, width=width)
    cv.imshow(WINDOW_NAME, img)
    while not ord("q") == cv.waitKey(200):
        pass
    cv.destroyAllWindows()


def resize(img, width: int = None, height: int = None, inter=cv.INTER_AREA):
    h, w = img.shape[:2]

    if width is None and height is None:
        raise ValueError("Specify either width or height.")

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img
