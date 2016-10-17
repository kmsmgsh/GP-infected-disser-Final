import numpy as np
import matplotlib.pyplot as plt
import coordinate as cr
import matplotlib.animation as animation
import generator_temp_transprob as gc
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy import optimize as op
import GP as gp
class Estimation:
    def __init__(self,record,geo,method="gradient"):
        '''
        record is a 2-dimensional numpy array which is n*d. n is number of people, d is the infectious time
        '''
        self.record=record
        self.geo=geo
        [self.days,self.num_people]=record.shape
        self.method=method
        self.DistanceMatrix=cr.DistanceMatrix(self.geo)
        self.change=self.changeMatrix()
        self.n=geo.shape[0]
        self.GPsize=self.n*(self.n-1)/2
    def transform_prob(self,state,gamma,BetaMatrix):
        """
        Special edition for Likelihood
        set the none infectious as 1
        transform the current state to next step:
        suspect(0)->infectious(1) with probability Lambda(state,beta)
        infectious(1)->removal(2) with probability np.exp(-gamma)
        returns the state after 1 day
        """
        p_S_to_I = self.Lambda(state,BetaMatrix)
        #now it is a num_people array
        p_I_to_R = 1-np.exp(-gamma)
        prob_transitions = np.zeros((self.num_people,))
        #if sum(p_S_to_I[p_S_to_I!=0])!=0:
        #prob_transitions[state==0] = p_S_to_I[p_S_to_I!=0]
        prob_transitions=p_S_to_I
        #prob_transitions[state==0] = p_S_to_I[p_S_to_I!=0]
        prob_transitions[state==1] = p_I_to_R
        prob_transitions[state==2] = 0
        return prob_transitions
    def Lambda(self,state,BetaMatrix):
        '''
        Lambda function returns a transform probability for every state to next step(S->I, I->R)
        This Lambda function is based on the "infected pressure" model which is differ from the previous one "Infected probability model".
        Infected pressure model is talking about that each infected people have a pressure to the surrounding suspected people, instead of
        the "probability at least one Infected object infect object j". The probability of j get infected in the end of the day is the 
        cumulative pressure for the infected objects surrouding j.
        For more detail, plz check "record 6.23.pdf" 2.1
        '''
        probInfect=np.zeros((self.num_people))
        probInfect[state==0] = BetaMatrix[:, state==1].sum(1)[state==0]
        probInfect=1-np.exp(-probInfect)
        return probInfect

    def changeMatrix(self):
        '''
        return a nday-1 * num_people matrix to show which state will change in next day
        The last day will not change so dimension is nday-1
        '''
        recordN=np.delete(self.record,0,0)#recordN is a matrix that have no first column for record 1,0,0,0,0,0
        change=np.vstack((recordN,self.record[self.days-1,:]))#make change is same dimension to self.record
        change=change-self.record#demonstrate the tranform procedure 
        change=np.delete(change,-1,0)#change is a matrix to record the change of state, which describe the change and result in record N
        #final row is useless so delete it
        return change

    def ProbabilityMatrix(self,BetaMatrix,gamma):
        Likeli=np.ones((self.num_people,1)).T
        for column in self.record:
            #print(column)
            #print("probability")
            k=self.transform_prob(column,gamma,BetaMatrix)
            #print(k)
            Likeli=np.vstack((Likeli,k))
            #print(Likeli)
        Likeli=np.delete(Likeli,-1,0)#delete the final row of the probability matrix(final row is no next generation so that the probaiblity matrix is 1 and 0 )
        Likeli=np.delete(Likeli,0,0)
        return Likeli

    def Likelihood(self,parameter,GP):
        gamma=parameter[-1]
        BetaMatrix=cr.BetaMatrix(self.DistanceMatrix,np.delete(parameter,-1),self.method)
        ####Add random  effect GP
        BetaMatrix=np.exp(np.log(BetaMatrix)+gp.LowerTriangularVectorToSymmetricMatrix(GP,BetaMatrix.shape[0]))
        probabilityMatrix=self.ProbabilityMatrix(BetaMatrix,gamma)
        change=self.change
        probabilityMatrix[change==0]=1-probabilityMatrix[change==0]
        loglikelihoodMatrix=np.log(probabilityMatrix)
        logLikelihood=loglikelihoodMatrix.sum(1).sum()
        return logLikelihood 
    def NonParametricGPLikelihood(self,GP,gamma):
        '''
        Likelihood for non-parametric model
        '''
        BetaMatrix=cr.BetaMatrix(self.DistanceMatrix,None,GP,"GaussianProcess")
        probabilityMatrix=self.ProbabilityMatrix(BetaMatrix,gamma)
        change=self.change
        probabilityMatrix[change==0]=1-probabilityMatrix[change==0]
        loglikelihoodMatrix=np.log(probabilityMatrix)
        logLikelihood=loglikelihoodMatrix.sum(1).sum()
        return logLikelihood
    def GaussinPriorNonparametricGP(self,gamma,GaussianProcess,GP):
        '''
        Not test
        '''
        return GaussianProcess.GPprior(GP)+self.NonParametricGPLikelihood(GP,gamma)


    def GammaPriorGeneralPosterior(self,parameter,GP ,i):
        from scipy.stats import gamma
        return np.log(gamma.pdf(parameter[i],0.001,scale=1/0.001))+self.Likelihood(np.array((parameter)),GP)
    
    def GaussianPriorGP(self,parameter,GaussianProcess,GP):
        '''
        Not fully test 
        but worked
        '''
        return GaussianProcess.GPprior(GP)+self.Likelihood(np.array((parameter)),GP)
        
    def GaussianStandardPriorGP(self,parameter,GaussianPriorGP,GP):
        from scipy.stats import multivariate_normal
        prior=multivariate_normal.logpdf(GP,mean=None,cov=GaussianPriorGP.CovarianceMatrix)
        prior2=GaussianPriorGP.GPprior(GP)
        Like=self.Likelihood(np.array((parameter)),GP)
        return prior+Like

    def PartialLikelihoodGamma(self,gamma,parameters=[0.4,0.1]):
        NONGP=np.zeros(self.GPsize)
        return self.Likelihood(np.append(parameters,gamma),NONGP)

'''
model1=gc.heteregeneousModel(100,0.4,0.3,10,True,False)
model1.Animate()
estimate=Estimation(model1.record,model1.geo,model1.phi)
r=np.linspace(0.1,0.7,200)
k=0.03*np.ones(200)
n= np.column_stack((r,k))
#for i in n:
#    print(estimate.Likelihood(i))]
k=[estimate.Likelihood(i) for i in n]
plt.plot(r,k)
k=0.4*np.ones(200)
n= np.column_stack((k,r))
k2=[estimate.Likelihood(i) for i in n]
plt.plot(r,k2)
plt.show()
#plt.show()
print("optimize!")
estimate.Likelihood(np.array((0.4,0.03)))

print("results")
#k=minimize(estimate.Likelihood,x0=np.array((0.001,0.03)),method="L-BFGS-B",bounds=np.array(([0.001,0.5],[0.001,0.9])))
#k=minimize(estimate.Likelihood,x0=0.001,method="BFGS",bounds=np.array((0.001,0.5)),tol=1e-190)
x0=[0.3,0.03]
#k=minimize(estimate.Likelihood,x0,method="BFGS")
#k=minimize(estimate.BetaLikelihoodQ,0.3,method="BFGS")
k= minimize(estimate.BetaLikelihoodQ,x0=0.3, method='L-BFGS-B')

k= minimize(estimate.BetaLikelihoodQ,x0=0.3, method='Nelder-Mead')
print(k)
#k=minimize_scalar(estimate.BetaLikelihoodQ)
bnds = np.array(([0.001,0.999],[0.001,0.999]))
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='Nelder-Mead')#Nelder-Mead can not handle with constrains, But at least works
x0 = [0.3,0.2]
k1=minimize(estimate.Likelihood, x0, method='BFGS')#BFGS do not work
k1=minimize(estimate.Likelihood, x0, method='Powell')#0.4086 and 73.70
k1=minimize(estimate.Likelihood, x0, method='CG')#return initial Value
k1=minimize(estimate.Likelihood, x0, method='COBYLA')#[ 0.40857675,  5.79367207]

print(k1)
#k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='Newton-CG')#Newton CG need jacobian, do not work
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='SLSQP')#do not work, return initial Value
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='TNC')# do not work, return initial Value
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='L-BFGS-B')# do not work, return initial Value
#op.fmin_bfgs(estimate.Likelihood,[0.3,0.02,])
#k1=op.brute(estimate.Likelihood, ((0, 1), (0, 0.999)))
#rranges = (slice(0, 1, 0.01), slice(0, 1, 0.01))
#k1=op.brute(estimate.Likelihood, rranges)
print(k1)
'''