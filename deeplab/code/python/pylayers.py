import sys, os
import numpy as np

import math
from scipy.ndimage import zoom
import theano
import theano.tensor as T

import cPickle, json

import numbers
import caffe

class SparseCutLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs 2 inputs!")

        from theano.tensor import maximum as max
        from theano.tensor import minimum as min

        N = [(0,1), (1,0), (1,1), (1,-1)]

        image = T.ftensor4()
        shape = image.shape
        X = shape[2]
        Y = shape[3]

        sigma = 0
        cnt = 0
        for dx, dy in N:
            i1 = image[:, :, max(0, dx):min(X+dx, X), max(0, dy):min(Y+dy, Y)]
            i2 = image[:, :, max(0,-dx):min(X-dx, X), max(0,-dy):min(Y-dy, Y)]
            sigma += T.sum((i1 - i2)**2, axis=(1,2,3), keepdims=True)
            cnt += (1-max(0, dx)+min(X+dx, X)) * (1-max(0, dy)+min(Y+dy, Y))
        sigma = sigma / cnt * 2

        labels = T.ftensor4()
        cut = 0

        for dx, dy in N:
            s1 = labels[:, :, max(0, dx):min(X+dx, X), max(0, dy):min(Y+dy, Y)]
            s2 = labels[:, :, max(0,-dx):min(X-dx, X), max(0,-dy):min(Y-dy, Y)]
            d = math.sqrt(dx * dx + dy * dy)

            i1 = image[:, :, max(0, dx):min(X + dx, X), max(0, dy):min(Y + dy, Y)]
            i2 = image[:, :, max(0,-dx):min(X - dx, X), max(0,-dy):min(Y - dy, Y)]
            delta = T.sum((i1 - i2) ** 2, axis=1, keepdims=True)

            cut -= T.sum(s1 * s2 * T.exp(- delta / sigma) / d)

        cut /= cnt

        self.forward_ = theano.function([labels, image], cut)
        self.backward_ = theano.function([labels, image], T.grad(cut, labels))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        labels = bottom[0].data
        data = bottom[1].data
        top[0].data[...] = self.forward_(labels, data)

    def backward(self, top, prop_down, bottom):
        labels = bottom[0].data
        data = bottom[1].data
        v = top[0].diff
        bottom[0].diff[...] = self.backward_(labels, data) * v


