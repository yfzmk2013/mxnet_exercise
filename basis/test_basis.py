#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np


def main():
    mx.random.seed(128)
    a = mx.nd.zeros((2, 3))
    b = mx.nd.ones((2, 3))
    c = mx.nd.full((2, 3), 7)
    d = mx.nd.random.normal(shape=(2, 2))
    print d
    p = np.random.rand(2, 4)  # 2行4列
    q = mx.nd.array(p)
    print type(p), type(q)
    print p, q
    print "end"


if __name__ == '__main__':
    main()
