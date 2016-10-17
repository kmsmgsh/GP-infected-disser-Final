import numpy as np 
import GP as gp 
import coordinate as cr 

geo=np.loadtxt("geo_uniform.txt")

DistanceMatrix=cr.DistanceMatrix(geo)
recordGP=np.loadtxt("GPTrueValueGradient30k20160828-021311.csv",delimiter=",")
record=np.array((0.4,0.1,0.4))


#gp.kernelFunctonPlot(DistanceMatrix,recordGP,record,"gradient")
gp.kernelFunctonPlot2(DistanceMatrix,recordGP,[0.4,0.1,0.3],"gradient")
gp.kernelFunctonPlotRebuild(DistanceMatrix,recordGP,[0.4,0.1,0.3],"gradient",[0.4,0.1,0.3])
gp.plotOneComponentGP(29999,1,recordGP)
