import cv2
import jpeg4py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from torch.utils.data.distributed import DistributedSampler


davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):

    if default_image_loader.use_jpeg4py is None:

        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False

        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)

    return opencv_loader(path)



default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:

        print(e)
        return None


def opencv_loader(path):
    try:
        im = cv2.imread(path, cv2.IMREAD_COLOR)

        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
        return None


def pil_loader(path):

    img = np.asarray(Image.open(path)).transpose((1, 0, 2))
    return img


def jpeg4py_loader_w_failsafe(path):
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            return None


def opencv_seg_loader(path):
    try:
        return cv2.imread(path)
    except Exception as e:
        print(e)
        return None


def imread_indexed(filename):
    im = Image.open(filename)
    ananotation = np.atleast_3d(im)[..., 0]
    return ananotation


def imwrite_indexed(filename, array, color_palette=None):

    if color_palette is None:
        color_palette = davis_palette
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception('it is not a 2d matrix')
    im = Image.fromarray(array)

    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

