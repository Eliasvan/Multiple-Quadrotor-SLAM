#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import os
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from gtsam import *



def vector(vec):
    return np.array([vec.coeff(i) for i in range(vec.rows())])

def matrix(mat):
    return np.array([[mat.coeff(i, j) for j in range(mat.cols())] for i in range(mat.rows())])

v2 = VectorXd(2)
print (vector(v2))
p2 = Point2(v2)
p2.print_()
print (vector(p2.vector()))

v3 = VectorXd(3)
print (vector(v3))
p3 = Point3(v3)
p3.print_()
print (vector(p3.vector()))

from math import pi
r2 = Rot2(pi/4)
r2.print_()
print (r2.theta())
print (matrix(r2.matrix()))

r3 = Rot3_quaternion(1/3**.5, 1/3**.5, 0., 1/3**.5)
r3.print_()
print (vector(r3.quaternion()))
print (matrix(r3.matrix()))

p = Pose2(1., 2., -1.)
p.print_()

pp = Pose3(r3, p3)
pp.print_()

s = Diagonal_Sigmas(v3)
s.print_("")
s2 = Isotropic_Sigma(3, 4.)
s2.print_("")

k = 1
f = PriorFactor_Pose2(k, p, s)
f.print_("")

ks = symbol('x', 1)
print (symbolChr(ks), symbolIndex(ks))
fs = PriorFactor_Pose2(ks, p, s)
fs.print_("")

g = NonlinearFactorGraph()
g.add(fs)
g.print_()

i = Values()
i.insert(ks, p)
i.print_()
i.exists(k)
i.exists(ks)

ps = LevenbergMarquardtParams()
ps.getVerbosityLM()
l = LevenbergMarquardtOptimizer(g, i, ps)
r = l.optimize()
r.print_()

m = Marginals(g, r)
cov = m.marginalCovariance(ks)
print (matrix(cov))

g.resize(0)
g.print_()

i.insert(k, p3)
i.print_()
Values(i.filter_Point3()).print_()
i.clear()
i.print_()
i.exists(ks)

NonlinearEquality_Pose2(k, p).print_()
BetweenFactor_Pose2(ks, k, p, s).print_("")

c = Cal3_S2(7., -5., 0.5, .25, .35)
c.print_()
print (matrix(c.K()))

c2 = Cal3DS2(7., -5., 0.5, .25, .35, 1., 0.4, .1, .15)
c2.print_()
print (matrix(c2.K()))
print (c2.k1(), c2.k2(), c2.p1(), c2.p2())

isam = NonlinearISAM()
g.add(fs)
i.insert(ks, p)
g.add(PriorFactor_Point3(k, p3, s2))
i.insert(k, p3)
isam.update(g, i)
isam.print_()
isam.printStats()
est = isam.estimate()
est.print_()
cov = isam.marginalCovariance(k)
print (matrix(cov))
isam.saveGraph("test_graph.dot")

isam2_params = ISAM2Params()
isam2_params.getFactorization()
isam2_params.factorization = Factorization.QR
isam2_params.getFactorization()
isam2 = ISAM2(isam2_params)
isam2.update(g, i)
isam2.update()
isam2.print_()
isam2.printStats()
est = isam2.calculateEstimate()
est.print_()
est = isam2.calculateBestEstimate()
est.print_()
cov = isam2.marginalCovariance(k)
print (matrix(cov))
