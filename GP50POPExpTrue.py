import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import Metropolis3 as mp3
import generator_temp_transprob as gc2
import likelihodPhi as lk2
import coordinate as cr
from scipy.stats import beta
from functools import partial
import GP as gp

population=50
model1=gc2.heteregeneousModel(population,[0.4,0.1,0.3],True,True)
model1.Animate()
estimate=lk2.Estimation(model1.record,model1.geo)
InitialGP=np.zeros(population*(population-1)/2)#zero GP and no transform
GPDoc=gp.GaussianProcess(estimate.DistanceMatrix,np.array((1,np.mean(estimate.DistanceMatrix))))
InitialGP=GPDoc.SampleForGP(np.zeros(population*(population-1)/2))
GPDoc=gp.GaussianProcess(estimate.DistanceMatrix,np.array((0.01,np.mean(estimate.DistanceMatrix))))
Metro=mp3.multiMetropolis(3000,None,[0.4,0.1,0.3],None,InitialGP,GPDoc,estimate.GaussianStandardPriorGP,"Change","c",1000)

gp.kernelFunctonPlot(model1.DistanceMatrix,Metro.recordGP,Metro.record,"gradient")
gp.kernelFunctonPlotRebuild(model1.DistanceMatrix,Metro.recordGP,Metro.record[0,:],"gradient",Metro.record[0,:],InitialGP)
Metro.printAcceptRateGP()
Metro.plotOneComponentGP(0)
Metro.plotOneComponentGP(1)
Metro.plotOneComponentGP(2)
Metro.plotOneComponentGP(3)
Metro.plotOneComponentGP(4)
Metro.plotOneComponentGP(5)
Metro.plotOneComponentGP(6)
plt.show()

gp.GPPlot(model1.DistanceMatrix,Metro.recordGP)
Metro.saveResult("TrueValueGradient3kRandomWalk","ConstantGradient3kRandomWalk")

