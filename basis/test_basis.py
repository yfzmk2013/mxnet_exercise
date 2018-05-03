#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
import torch
import tensorflow as tf
def main():
    print "start!"
    mx.random.seed(128)
    a = mx.nd.zeros((2, 3), dtype=np.int32)
    b = mx.nd.ones((2, 3))
    c = mx.nd.full((2, 3), 7)
    print a, b, c
    print a.dtype, b.dtype
    d = mx.nd.random.normal(shape=(2, 2))
    print d
    # p = np.random.rand(2, 4)  # 2行4列
    p = np.eye(5, 5)
    q = mx.nd.array(p)#numpy ndarray -> NDAray
    qs = mx.nd.exp(q)
    qt=qs.asnumpy() # NDarray->numpy ndarry
    print 'p.shap = ' + str(p.shape)
    print 'q.shap = ' + str(q.shape)
    print type(p), type(q)
    print p, q, qs,qt
    print type(qt)

    p1 = torch.rand(4, 3)
    n1=p1.numpy()#ptensor->numpy ndarry
    p2 = torch.from_numpy(n1) #numpy ndarry->numpy

    print type(p1),type(n1),type(p2)

    print "end!"


if __name__ == '__main__':
    main()
