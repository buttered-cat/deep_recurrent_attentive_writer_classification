What's This?
---------------

This is a python3 + tensorflow1.2 implementation of [Deep Recurrent Attentive Writer (DRAW)](https://arxiv.org/abs/1502.04623) model as a classifier. It supports RGB images of arbitrary size as input, with paddding added in each batch to form a tensor, and uses a softmax layer on encoder state sequence to classify inputs. The code is developed with PyCharm.

I've also tried processing the images one at a time as an alternative to padding, which is slower. If you are open minded and want to try something fresh, you can checkout branch `ckpt_before_performance_optimization` for the code.

The code is based on [kvfrans/draw-color](https://github.com/kvfrans/draw-color) (which is based on [ericjang/draw](https://github.com/ericjang/draw)), the major difference being that 1) It only supports square images of the same size as input. 2) it doesn't have a classification layer; 3) it uses python2 & tensorflow < 1.2; I've also modified some details which I perceived as programming fault.

Dependencies
---------------

* Tensorflow >= 1.2.1
* python >= 3.5
* numpy
* scipy
* pickle
* pillow


TODO's and Known Issues
---------------

This model is originally proposed for [an image classification contest held by Baidu](http://js.baidu.com/), but sadly I hadn't had time to finish it by the time it was due. I'm still working on it and the list will be updated accordingly.

1. Currently the VAE part and the classification part can't be trained separately, making it hard to use unlabeled data.
3. I'm using the sequence of encoder states as classifier input. Might as well use latent code sequence?
4. Validation + early stopping.
5. Variable attention window aspect ratio?
6. Resume training.
7. Tensorboard functions.
