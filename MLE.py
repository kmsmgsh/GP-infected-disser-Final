import numpy as np 
import generator_temp_transprob as gc
import matplotlib.pyplot as plt
import time
import likelihodPhi as lk 
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
'''
This code is working for get the MLE of the likelihood
and draw the likelihood function for gamma, and contour plot for beta_0 and phi
'''
population=50
parameter=[0.4,0.1,0.3]
def mleGamma(Likelihood,parameterEXP=np.array((0.4,0.1))):
    n=np.linspace(0,1,200)
    parameters=np.zeros((3,200))
    parameters[0:,]=parameterEXP[0]
    parameters[1:,]=parameterEXP[1]
    parameters[2:,]=n
    #plt.plot(n,[np.exp(Likelihood.Likelihood(column,NONGP)) for column in parameters.T])   
    #plt.plot(n,[np.exp(Likelihood.Likelihood([0.4,0.1,i],NONGP)) for i in n])
    values=[np.exp(Likelihood.PartialLikelihoodGamma(i,[0.4,0.1])) for i in n]
    plt.plot(n,values)
    print(n[np.argmax(values)])
    plt.show()
def MLE(Likelihood):
    NONGP=np.zeros(Likelihood.GPsize)
    def negativeLikelihood(parameter,GP):
        return -Likelihood.Likelihood(parameter,GP)
    def negativePartialLikelihood(gamma):
        return -Likelihood.PartialLikelihoodGamma(gamma)
    k=minimize(negativeLikelihood,x0=np.array((0.001,0.03,0.03)),method="Nelder-Mead",args=NONGP)
    print(k)
    k=minimize_scalar(negativePartialLikelihood,bounds=(0,0.8))
    print(k)
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
def countourLikelihood(Likelihood):
    X = np.arange(0.01, 0.8, 0.005)
    Y = np.arange(0.01, 0.3, (0.3-0.01)*0.005/(0.8-0.01))
    NONGP=np.zeros(Likelihood.GPsize)
    #X, Y = np.meshgrid(X, Y)
    def negativeLikelihood(parameter,GP):
        return -Likelihood.Likelihood(parameter,GP)
    Z=np.zeros((X.size,Y.size))
    for j in range(X.size):
        for i in range(Y.size):
            Z[i,j]=np.exp(-negativeLikelihood(np.array((X[i],Y[j],0.3)),NONGP))
            #Z[i,j]=np.exp(Likelihood.Likelihood([X[i],Y[j],0.3],NONGP))
    X, Y = np.meshgrid(X, Y)
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #ax.plot_surface(X,Y,Z, rstride=1, cstride=1, color='b')
    #ax.set_zlim(0.1, 0.501)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.contour(X,Y,Z)
    plt.show()
 
if __name__ == "__main__":
    population=50    
    distanceMethod="gradient"
    if distanceMethod=="gradient" :
        parameter=[0.4,0.1,0.3]
    elif distanceMethod=="powerlaw":
        parameter=[200,0.1,1.5,0.3]
    model1=gc.heteregeneousModel(population,parameter,True,True,distanceMethod)
    #model1.Animate()
    Likelihood=lk.Estimation(model1.record,model1.geo)
    mleGamma(Likelihood)
    MLE(Likelihood)
    countourLikelihood(Likelihood)