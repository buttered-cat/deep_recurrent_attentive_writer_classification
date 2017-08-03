import scipy.misc
import numpy as np
import random
import tensorflow as tf
import _pickle as cPickle


# check https://stackoverflow.com/questions/38966533/different-image-sizes-in-tensorflow-with-batch-size-1
def get_image(image_path):
    with open(image_path, mode='rb') as file:
        return tf.image.decode_jpeg(file.read(), channels=3)
        # [height, width, channels] tensor

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    # take center
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def imread(path):
    readimage = scipy.misc.imread(path, mode="RGB").astype(np.float)
    return readimage

def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):        # lay images row by row
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def ims(name, img):
    # print img[:10][:10]
    scipy.misc.toimage(img, cmin=0, cmax=1).save(name)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
