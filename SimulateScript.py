import numpy as np 
import generator_temp_transprob as gc
import matplotlib.pyplot as plt
import time

population=50
num_simulations=600
'''
This code is about exploratory analysis about the link between the parameter with the epidemics simulation results.
Explore the relationship between the I(t) with the input parameter
'''
#model=gc.heteregeneousModel(population,[0.4,0.1,0.3],True,False)
#a=model1.cout1

def SimulationInfectedPeople(population=50,num_simulations=600,parameter=[0.4,0.1,0.3],distance="gradient",filename="General"):
    results = np.array([gc.heteregeneousModel(population,parameter,True,False,distance).cout1 for _ in range(num_simulations)])
    daysEachSimulation=[len(i) for i in results]                    #for auto stop method, the dimensions of the infected people everyday is not equal
    daysEachSimulationArray=np.array(daysEachSimulation)            #In order to plot the data, maybe we should make it into same dimension with 0 append
    rank=max(daysEachSimulationArray)
    ResultsWithSameDimension=np.zeros([len(results),rank])
    for i in range(0,len(results)):
        for j in range(len(results[i])):
            ResultsWithSameDimension[i][j]=results[i][j]
    plt.plot(ResultsWithSameDimension.T,'b')
    plt.show()
    timestr=time.strftime("%Y%m%d-%H%M%S")
    np.savetxt("SeveralSimulationResult"+filename+timestr+".csv", ResultsWithSameDimension, delimiter=",")
def SimulationFinalSize(population=50,num_simulations=600,parameter=[0.4,0.1,0.3],distance="gradient",filename="General"):
    results = np.array([gc.heteregeneousModel(population,parameter,True,False,distance).cout2 for _ in range(num_simulations)])
    results=np.array(results)
    plt.hist(results)
    plt.show()
    timestr=time.strftime("%Y%m%d-%H%M%S")
    np.savetxt("FinalSizeSimulation"+filename+timestr+".csv", results, delimiter=",")
if __name__ == "__main__":
    SimulationInfectedPeople(50,600,[200,0.1,1.5,0.3],"powerlaw","Power")
    SimulationFinalSize(50,600,[200,0.1,1.5,0.3],"powerlaw","Power")
    #model1=gc.heteregeneousModel(population,[0.4,0.1,0.3], True ,True)
    #model1.Animate()
