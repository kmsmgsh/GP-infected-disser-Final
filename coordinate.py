import numpy as np
import matplotlib.pyplot as plt
import GP as gp
'''
class coordinate:
    def __init__(self,x=0.0,y=0.0):
        self.x=x
        self.y=y
a=coordinate()
'''
'''
Changelog 2016/7/28
Change powerlaw method for beta matrix as three input parameter:beta0,phi,omega
and add the input test and error output
'''
def geodata(num_people,method="uniform",xbound=100.0,ybound=100.0,history=False):
    """
    Generator a n*2 array, which is the x-coordinate and y-coordinate
    in each row,for n people
    The method equal to uniform means each point in 2-D surface is uniform
    distributed
    The method equal to cluster means points x01,x02,...x0m,are normally distributed
    around a point x0, x0 is uniform in this surface
    """
    
   

    if method=="uniform":
        if history==True:
            try:
                 population=np.loadtxt("geo_uniform.txt")
                 return population
            except FileNotFoundError:
                print("no file found, auto create new file")
        population=np.zeros((num_people,2))
        population[:,0]=np.random.uniform(0,xbound,num_people)
        population[:,1]=np.random.uniform(0,ybound,num_people)
        print(population)
        plt.scatter(population[:,0],population[:,1])
        plt.show()
        np.savetxt("geo_uniform.txt",population)
        return population
    if method=="cluster":
        if history==True:
            try:
                 population=np.loadtxt("geo_cluster.txt")
                 return population
            except FileNotFoundError:
                print("no file found, auto create new file")
        population=np.zeros((num_people,2))
        
        num_cluster=int(np.random.uniform(0.1,1)*10+1)
        # the number of cluster follow uniform 1,2,...10
        # problem: how to scientifically define the number of cluster?
        people_set=cluster_people(num_people,num_cluster)
        center_point=np.zeros((num_cluster,2))
        center_point[:,0]=np.random.uniform(0,xbound,num_cluster)
        center_point[:,1]=np.random.uniform(0,ybound,num_cluster)
        ######################################################
        iter=0
        cout=0
        for i in people_set:    
            for j in range(i):
                population[iter,:]=np.random.multivariate_normal(center_point[cout,:],25*np.identity(2))
                    #problem: how to detemine the covariance matrix?
                    #btw: 25 is only pick randomly.1 is too small in graph
                iter+=1
            cout+=1
        ######################################################
        #plt.scatter(population[0,0],population[0,1],cmap=3)
        #plt.scatter(population[1:num_people-1,0],population[1:num_people-1,1])
        #plt.show()
        #maybe I should put the show section into one part(rather than in each if)
        np.savetxt("geo_cluster.txt",population)
        return population        
def cluster_people(num_people,num_cluster):
    '''
        The following code is try to split the number of poeple into num_cluster
        sets, which means n1 people in cluster1, n2 people in cluster2,..., and
        n1+n2+...+n3=num_people
        This function return an array which is num_people*1, each element is the
        number of people in each cluster
    '''
    if num_cluster==1:
        return num_people
    first=int (np.random.uniform(1,num_people-num_cluster))
    return np.hstack((first,cluster_people(num_people-first,num_cluster-1)))
    '''     Following is  Test code
    population=geodata(100,"uniform",100,100)
    beta=0.3

    for i in range(100):
    distance1=np.linalg.norm(population[1,:]-population[i,:])
    print("distance is")
    print(distance1)
    print("probability is")
    print(beta*np.exp(-(distance1)/20))
    '''
#population=geodata(100,"cluster",100,100,True)
#print(population)
def DistanceMatrix(geo):
    num_people=geo.shape[0]
    DistanceMatrix=np.zeros([num_people,num_people])
     #self.BetaMatrix=[for i,j in geo[,:]]
    X=geo
    DistanceMatrix=-2 * np.dot(X, X.T) + np.sum(np.square( X), 1).reshape(num_people, 1) + np.sum(np.square( X), 1).reshape(1, num_people)
    for i in range(0,num_people):
        DistanceMatrix[i,i]=0
    DistanceMatrix=np.sqrt(DistanceMatrix)
    return DistanceMatrix 
def BetaMatrix(DistanceMatrix,parameter,method="gradient",GP=None):   
    if method=="gradient":
        beta0=parameter[0]
        phi=parameter[1]
        BetaMatrix=beta0*np.exp(-DistanceMatrix*phi)
        #K takes distance -arg paratemeters->
        #returns 
        #self.BetaMatrix=np.array(BetaMatrix)
        return BetaMatrix
    elif method=="powerlaw":
        parameter=np.array(parameter)
        if parameter.size!=3:
            raise ValueError("powerlaw method need 3 arguments as beta0,phi,omega")
        beta0=parameter[0]
        phi=parameter[1]
        omega=parameter[2]
        BetaMatrix=beta0*1/(phi**2+DistanceMatrix**2)**omega
        return BetaMatrix
    elif method=="GaussianProcess":
        m=GP.size
        n=(1+np.sqrt(1+8*m))/2
        BetaMatrix=gp.LowerTriangularVectorToSymmetricMatrix(GP,n)
        return np.exp(BetaMatrix)
    elif method=="gradientGaussianProcess":
        beta0=parameter[0]
        phi=parameter[1]
        n=(1+np.sqrt(1+8*GP.size))/2
        BetaMatrix=beta0*np.exp(-DistanceMatrix*phi)
        BetaMatrix=BetaMatrix*np.exp(gp.LowerTriangularVectorToSymmetricMatrix(GP,n))
        return BetaMatrix
    elif method=="powerlawGaussianProcess":
        parameter=np.array(parameter)
        beta0=parameter[0]
        phi=parameter[1]
        omega=parameter[2]
        n=(1+np.sqrt(1+8*GP.size))/2
        BetaMatrix=beta0*1/(phi**2+DistanceMatrix**2)**omega
        BetaMatrix=BetaMatrix*np.exp(gp.LowerTriangularVectorToSymmetricMatrix(GP,n))
        return BetaMatrix
def plotcoor(geo):
    plt.scatter(geo[:,0],geo[:,1])
    #plt.figure()
