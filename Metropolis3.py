import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import GP as gp
import time
import csv
#plt.ion()
#plt.style.use('ggplot')
class Metropolis:
    def OnestepMetropolis(self,density,theta,sigma,GP,i):
        theta_star=np.exp(np.random.normal(np.log(theta[i]), sigma))
        thetastar=theta.copy()
        thetastar[i]=theta_star
        #Accept the new beta value with the probability f(beta_start)/f(beta)
        p=min(np.exp(density(thetastar,GP)-density(theta,GP)),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,theta_star]
            #count the number of accept
        else:
            return [0,theta[i]]
    def OnestepMetropolisGP(self,densityGP,parameter,GaussProcess,CurrentGP):
        newGP=GaussProcess.SampleForGP(CurrentGP)
        #Accept the new beta value with the probability f(beta_start)/f(beta)
        lognew=densityGP(parameter,GaussProcess,newGP)
        logold=densityGP(parameter,GaussProcess,CurrentGP)
        mid=lognew-logold
        p=min(np.exp(mid),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,newGP]
            #count the number of accept
        else:
            return [0,CurrentGP]

        
class multiMetropolis(Metropolis):
    def __init__(self,IterNa,density,initial_parameter,sigma,initialGP,GaussianProcess=None,GPdensity=None,GPmode="c",parameterMode="m",AdaptiveConstant=1000):
        self.IterNa=IterNa
        self.initial_parameter=np.array(initial_parameter)
        self.dimension=self.initial_parameter.size
        self.density=density
        self.sigma=sigma
        self.initialGP=initialGP
        self.GaussianProcess=GaussianProcess
        self.GPmode=GPmode
        self.GPdensity=GPdensity
        self.parameterMode=parameterMode
        self.AdaptiveConstant=AdaptiveConstant
        self.Mainprocess()
    def Mainprocess(self):
        parameter=self.initial_parameter
        record=parameter
        Accept=np.zeros((self.IterNa,self.dimension))
        GP=self.initialGP
        self.AcceptGP=np.zeros(self.IterNa)
        self.recordGP=GP
        self.CovUpdate=gp.UpdatingCov(GP.size)
        f  = open('GP.csv', 'w')
        f2 = open('wP.csv', 'w')
        writer=csv.writer(f,delimiter=',')
        writer2=csv.writer(f2,delimiter=',')
        for i in range(0,self.IterNa):
            if self.parameterMode!="c":
                '''
                if parameterMode is constant, the parameter do not move, just avoid the MH algorithm
                value is as the initialValue
                the parameter part is as the mean of GP
                '''
                for j in range(0,self.dimension):
                    result=self.OnestepMetropolis(self.density[j],parameter,self.sigma[j],GP,j)
                    Accept[i,j]=result[0]
                    parameter[j]=result[1]
            #update recover rate sigma
            '''j=self.dimension-1
            result=self.OnestepMetropolisGPAdaptive(j,self.density[j],parameter,self.sigma[j],GP,j)
            Accept[i,j]=result[0]
            parameter[j]=result[1]
            '''
            record=np.vstack((record,parameter))
            writer2.writerow(parameter)
            
            if self.GPmode!="c":
                resultGP=self.OnestepMetropolisGPAdaptive(i,self.GPdensity,parameter,self.GaussianProcess,GP)
                self.AcceptGP[i]=resultGP[0]
                self.recordGP=np.vstack((self.recordGP,resultGP[1]))
                GP=resultGP[1]
                writer.writerow(GP)
            ##############################################
            #self.AdaptiveCovariance=gp.updateCor(GP)#####
            ##############################################
        self.record=record
        self.Accept=Accept
        #if self.GPmode!="c":
            #self.recordGP=recordGP
            #self.AcceptGP=AcceptGP
        
        f.close()
        f2.close()
    
    
    def OnestepMetropolisGPAdaptive(self,j,densityGP,parameter,GaussProcess,CurrentGP):
        '''
        The adaptive MCMC method. Change the Covariance Matrix everytime
        Use the empirical covariance for the MCMC output

        
        Monte Carlo not Markov Chain
        MCnMC XD

        '''    

        if j<self.AdaptiveConstant:
            #if iterative<200 then do the sample update
            newGP=GaussProcess.StandardSampleForGP(CurrentGP)
        else:
            #if j>200 then do the adaptive Sampler
            newGP=gp.AdaptiveSampleForGP(CurrentGP,self.CovUpdate.getOnlineCov())
            #newGP=gp.SampleForGPCov(CurrentGP,np.cov(self.recordGP.T))
            #newGP=GaussProcess.AdaptiveSampleForGPSpecial(CurrentGP,np.cov(self.recordGP.T))
            #newGP=GaussProcess.AdaptiveSampleForGPSpecial(CurrentGP,self.CovUpdate.getOnlineCov())
        #---->Seems we need online update cholesky decomposition  (/ﾟДﾟ)/ 
        #the update for the covariance is outside this function!
       
        #need rebuild the sample with History data
        #Something like newGP=GaussProcess.SampleForGP(CurrentGP,self.recordGP)
        #Then the following is the same

        #Accept the new beta value with the probability f(beta_start)/f(beta)
        lognew=densityGP(parameter,GaussProcess,newGP)
        logold=densityGP(parameter,GaussProcess,CurrentGP)
        mid=lognew-logold
        p=min(np.exp(mid),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        #print(p)
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            self.CovUpdate.updateOnlineCov(newGP)
            return [1,newGP]
            #count the number of accept
        else:
            self.CovUpdate.updateOnlineCov(CurrentGP)
            return [0,CurrentGP]

    def showplot(self,i):
        plt.clf()
        plt.plot(range(self.IterNa+1),self.record[:,i])
        plt.plot(range(200,self.IterNa),self.record[200:self.IterNa,i])
        plt.show()
        #plt.figure()
        plt.hist(self.record[:,i],bins=50)
        #plt.figure()

        plt.show()
    
    def plotcountour(self,i,j):
        plt.clf()
        plt.plot(self.record[:,i],self.record[:,j], 'ko', alpha=0.4)
        plt.show()
        plt.plot(self.record[:,i],self.record[:,j])
        plt.show()
    def printall(self,i):
        print("Accept rate is")
        print(sum(self.Accept[:,i])/self.IterNa)
        print("Mean is")
        print(np.mean(self.record[200:self.IterNa,i]))
        print("Variance is")
        print(np.cov(self.record[200:self.IterNa,i]))
    def printAcceptRateGP(self):
        print("Accept rate is")
        ak=sum(self.AcceptGP)/self.IterNa
        print(ak)
        f=open("accept.txt",'w')
        f.write(np.array_str(ak))
        f.close()
    def plotOneComponentGP(self,i):
        plt.plot(range(self.IterNa+1),self.recordGP[:,i])
    def saveResult(self,filenameGP,filenameParameter):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.savetxt("GP"+filenameGP+timestr+".csv", self.recordGP, delimiter=",")
        print("GP results have successfully saved!")
        np.savetxt("parameter"+filenameParameter+timestr+".csv",self.record,delimiter=",")
        print("parameter results have successfully saved!")
class GaussianProcessMetropolis(Metropolis):
    '''
    This is the kernel function with only GP update but with some prior infor as gradient(exponential)
    '''
    def _init_(self,IterNa,initialGP,Gaussianprocess,GPdensity,baseline):
        self.IterNa=IterNa
        self.GPdensity=GPdensity
        self.GaussianProcess=GaussianProcess
        self.initialGP=initialGP
        self.baseline=baseline
        self.Dimension=initialGP.size
        self.populationsize=(1+np.sqrt(1+8*self.Dimension))/2
        self.Mainprocess()
    def Mainprocess(self):
        GP=self.initialGP
        AcceptGP=np.zeros(self.IterNa)
        recordGP=GP
        for i in range(self.IterNa):
            resultGP=self.OnestepMetropolisGP(self.GPdensity,parameter,self.GaussianProcess,GP)
            AcceptGP[i]=resultGP[0]
            recordGP=np.vstack((recordGP,resultGP[1]))
        