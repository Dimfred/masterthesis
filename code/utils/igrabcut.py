# main
import cv2 as cv
import numpy as np

# utilities
from pathlib import Path
from typing import Union, Tuple
from copy import deepcopy
import os

# TODO remove this import
import utils


class Hotkeys:
    crop = ord("c")
    background = ord("b")
    probably_background = ord("1")
    # TODO
    # probably_background = ord("B")
    foreground = ord("f")
    probably_foreground = ord("2")
    # TODO
    # probably_foreground = ord("B")

    inc_brush = ord("=")
    dec_brush = ord("-")
    grabcut = ord("g")
    mask_generator = ord("a")

    save = ord("s")
    delete = ord("d")
    close = ord("q")
    reset = ord("r")


class colors:
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    black = (0, 0, 0)
    # TODO
    pink = (247, 7, 203)[::-1]
    white = (255, 255, 255)
    cyan = (7, 163, 247)[::-1]


class Mask:
    bg = 0
    fg = 1
    probably_bg = 2
    probably_fg = 3


class Utils:
    pass


def store_cursor_pos(func):
    def decorator(self, event, x, y, flags, params):
        # store cursor position in the application
        self.app._cursor_pos = (x, y)
        # call the mouse_cb
        func(self, event, x, y, flags, params)

    return decorator


class AppState:
    def __init__(self, app: "IGrabcut"):
        self.name = "NOTIMPLEMENTED"

        self.app = app

        self._empty_cb = lambda *args, **kwargs: None

    def set_mouse_cb(self, cb):
        if cb is None:
            cv.setMouseCallback(self.app.inwin_name, self._empty_cb)
            return

        cv.setMouseCallback(self.app.inwin_name, cb)


class Crop(AppState):
    def __init__(self, app: "IGrabcut"):
        super().__init__(app)
        self.name = "crop"

        # upper left corner, lower right corner
        self.p1, self.p2 = None, None

        self.set_mouse_cb(self.mouse_cb)

        self.drawing = False
        self.done = False

    def __call__(self):
        if self.done:
            return self.app.prev_app_state

        return self

    @store_cursor_pos
    def mouse_cb(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.init_rect(x, y)
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            self.crop()
            self.done = True
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.update_rect(x, y)

    def init_rect(self, x, y):
        self.p1 = (x, y)
        self.p2 = (x, y)
        self.render()

    def update_rect(self, x, y):
        self.p2 = (x, y)
        self.render()

    def crop(self):
        img = self.app.img

        x1, y1 = self.p1
        x2, y2 = self.p2

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # region of interest; the inside of the crop
        roi = img[ymin:ymax, xmin:xmax]

        # everything background, except roi
        img = np.zeros_like(img)
        img[y1:y2, x1:x2] = roi

        self.app.img = img

        if self.app._mask is None:
            self.app._mask = np.zeros(img.shape[:2], dtype=np.uint8)

        self.app._mask[y1:y2, x1:x2] = Mask.probably_fg
        self.app._crop = (x1, y1, x2 - x1, y2 - y1)

        self.render()

    def render(self):
        img = self.app.img
        cv.rectangle(img, self.p1, self.p2, colors.black)
        self.app._show_img = img


class IncreseBrush(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "increase brush"

    def __call__(self):
        self.app._brush_size += 2
        return self.app.prev_app_state


class DecreaseBrush(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "decrease brush"

    def __call__(self):
        # TODO maybe config
        if self.app._brush_size > 1:
            self.app._brush_size -= 2

        return self.app.prev_app_state


class DrawGround(AppState):
    def __init__(self, app, color, value, Derived):
        super().__init__(app)
        self.color = color
        self.value = value
        self.Derived = Derived
        self.drawing = False

        self.set_mouse_cb(self.mouse_cb)

        if self.app._mask is None:
            self.app._mask = np.zeros(self.app._img.shape[:2], dtype=np.uint8)

    def __call__(self):
        return self

    @store_cursor_pos
    def mouse_cb(self, event, x, y, *args):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.draw(x, y)
            self.render()
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            self.render()
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.draw(x, y)
            self.render()

    def draw(self, x, y):
        # TODO thickness?
        self.app.img = cv.circle(
            self.app._img, (x, y), self.app._brush_size // 2, self.color, -1
        )
        self.app._mask = cv.circle(
            self.app._mask, (x, y), self.app._brush_size // 2, self.value, -1
        )

    def render(self):
        self.app._show_img = self.app.img


class DrawForeground(DrawGround):
    def __init__(self, app):
        super().__init__(app, colors.blue, Mask.fg, DrawForeground)
        self.name = "foreground"


class DrawProbablyForeground(DrawGround):
    def __init__(self, app):
        super().__init__(app, colors.cyan, Mask.probably_fg, DrawProbablyForeground)
        self.name = "probably foreground"


class DrawBackground(DrawGround):
    def __init__(self, app):
        super().__init__(app, colors.red, Mask.bg, DrawBackground)
        self.name = "background"

class DrawProbablyBackground(DrawGround):
    def __init__(self, app):
        super().__init__(app, colors.pink, Mask.bg, DrawProbablyBackground)
        self.name = "probably background"


class Reset(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "reset"

    def __call__(self):
        self.app.img = self.app._original_img.copy()
        self.app._show_img = self.app.img

        return BaseState(self.app)


class Grabcut(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "grabcut"

    def __call__(self):
        if self.app._mask is None:
            print("No mask set, either crop the img, or draw bg / fg.")
            return self.app.prev_app_state

        mask, bg_model, fg_model = cv.grabCut(
            self.app._original_img.copy(),
            self.app._mask,
            self.app._crop,
            self.app._bg_model,
            self.app._fg_model,
            5,
            cv.GC_INIT_WITH_MASK,
        )

        # TODO what is happening here? seems kinda wrong
        # mask = np.uint8(np.where((mask == Mask.bg) | (mask == Mask.probably_bg), 0, 1))
        # mask = np.logical_or(mask[mask == Mask.fg], mask[mask == Mask.probably_fg])
        mask = np.uint8((mask == Mask.fg) | (mask == Mask.probably_fg))
        mask = mask[..., np.newaxis]

        # store mask in the app
        self.app._saved_mask = mask.copy()

        self.app._fg_img = self.app._original_img * mask
        self.app._fg_img[np.logical_not(mask[:, :, 0])] = colors.red


        return self.app.prev_app_state if not self.app.auto_save else Save(self.app)


class Close(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "close"

    def __call__(self):
        cv.destroyAllWindows()
        return self


class GenerateMask(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "mask generator"

    def __call__(self):
        if self.app.mask_generator is None:
            print("Please set the mask_generator.")
            return self.app.prev_app_state

        # generate mask from user defined function
        mask = self.app.mask_generator(self.app.img)

        self.app._mask = mask
        # TODO only if img has 3 channels
        self.app._img = self.app._img * (mask != 0)[..., np.newaxis]
        self.app._show_img = self.app.img

        return self.app.prev_app_state


class Save(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "save"

    def __call__(self):
        basename = os.path.basename(self.app._img_path)
        name, ext = os.path.splitext(basename)

        # TODO name into init
        path = Path(self.app.output_dir) / f"{name}_fg_mask{ext}"
        cv.imwrite(str(path), self.app._saved_mask)
        print(f"Saving to {path}.")

        return self.app.prev_app_state


class Delete(AppState):
    def __init__(self, app):
        super().__init__(app)
        self.name = "delete"

    def __call__(self):
        # TODO merge with save
        basename = os.path.basename(self.app._img_path)
        name, ext = os.path.splitext(basename)

        path = Path(self.app.output_dir) / f"{name}_fg{ext}"
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted {path}.")
        else:
            print(f"{path} does not exist.")

        return self.app.prev_app_state


class BaseState(AppState):
    """ State to switch between states """

    _states = {
        Hotkeys.crop: Crop,
        Hotkeys.inc_brush: IncreseBrush,
        Hotkeys.dec_brush: DecreaseBrush,
        Hotkeys.grabcut: Grabcut,
        Hotkeys.foreground: DrawForeground,
        Hotkeys.background: DrawBackground,
        Hotkeys.reset: Reset,
    }

    def __init__(self, app):
        super().__init__(app)
        self.name = "base"

        self.set_mouse_cb(self.mouse_cb)

    def __call__(self):
        return self.app.app_state

    @store_cursor_pos
    def mouse_cb(self, *args):
        pass


class IGrabcut:
    # fmt: off
    _states = {
        Hotkeys.grabcut: Grabcut,
        Hotkeys.crop: Crop,
        Hotkeys.foreground: DrawForeground,
        Hotkeys.probably_foreground: DrawProbablyForeground,
        Hotkeys.background: DrawBackground,
        Hotkeys.probably_background: DrawProbablyBackground,
        Hotkeys.mask_generator: GenerateMask,

        Hotkeys.inc_brush: IncreseBrush,
        Hotkeys.dec_brush: DecreaseBrush,

        Hotkeys.save: Save,
        Hotkeys.delete: Delete,
        Hotkeys.reset: Reset,
        Hotkeys.close: Close,
    }
    # fmt: on

    def __init__(
        self,
        output_dir: Union[Path, str],
        auto_save: bool = True,
        inwin_name="input",
        outwin_name="output",
        waitkey_delay=1,
    ):
        self.app_state = None
        self.waitkey_delay = waitkey_delay

        # windows
        self.inwin_name = inwin_name
        self.outwin_name = outwin_name

        # output
        self.auto_save = auto_save
        self.output_dir = output_dir

        # TODO naming function

        # images
        self._img_path = None
        self._original_img = None
        self._img = None
        self._show_img = None
        self._fg_img = None
        # TODO ?
        self._bg_img = None

        # ui
        self._cursor_pos = (0, 0)
        self._brush_size = 9

        # grabcut
        self.mask_generator = None
        self._mask = None
        self._crop = None
        self._bg_model = np.zeros((1, 65), dtype=np.float64)
        self._fg_model = np.zeros((1, 65), dtype=np.float64)

    def imread(self, path: Union[Path, str], resize: int, channels: int = 3):
        self._img_path = str(path)

        color_type = cv.IMREAD_GRAYSCALE if channels == 1 else None
        img = cv.imread(self._img_path, color_type)
        img = utils.resize_max_axis(img, resize)

        self._img = img
        self._original_img = self.img
        self._show_img = self.img
        self._fg_img = self.img

    def run(self, path: Union[Path, str], resize=1000):
        self.imread(path, resize)

        cv.namedWindow(self.outwin_name, cv.WINDOW_AUTOSIZE)
        cv.namedWindow(self.inwin_name, cv.WINDOW_AUTOSIZE)

        self.app_state = BaseState(self)

        while self.app_state.name != "close":

            img = self._show_img.copy()
            img = self.render_state(img)
            img = self.render_brush(img)
            img = self.render_cross(img)

            cv.imshow(self.inwin_name, img)
            cv.imshow(self.outwin_name, self._fg_img)

            # apply the current state
            self.app_state = self.app_state()

            key = self.pressed_key
            if key not in self._states:
                continue

            State = self._states[key]
            # some states restore the state after they we're applied, hence store
            # the old one
            self.prev_app_state = deepcopy(self.app_state)

            # initialize the new state
            self.app_state = State(self)

    @property
    def pressed_key(self):
        return 0xFF & cv.waitKey(self.waitkey_delay)

    @property
    def img(self):
        return self._img.copy()

    @img.setter
    def img(self, other):
        self._img = other

    # TODO all into seperate renderer
    def render_state(self, img):
        img = cv.putText(
            img,
            self.app_state.name,
            (10, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            colors.red,
            2,
            cv.LINE_8,
        )
        return img

    def render_brush(self, img):
        img = cv.circle(
            img, self._cursor_pos, self._brush_size // 2, colors.red, 2, cv.LINE_8
        )
        return img

    def render_cross(self, img):
        h, w = img.shape[:2]

        xcur, ycur = self._cursor_pos

        left, right = (0, ycur), (w - 1, ycur)
        img = cv.line(img, left, right, colors.red, 1)

        top, bot = (xcur, 0), (xcur, h - 1)
        img = cv.line(img, top, bot, colors.red, 1)

        return img
