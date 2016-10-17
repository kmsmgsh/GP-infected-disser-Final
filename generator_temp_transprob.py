import numpy as np
import matplotlib.pyplot as plt
import coordinate as cr
import matplotlib.animation as animation
import GP as gp
import time
'''
Changelog for generator_temp
Use another way to make transition probability in order to avoid potential problem for powerlaw distance decay function
'''
'''
A new logic for saving the epidemics result:
When ever using the animation function to see the result of the epidemics, then the result have its value to be saved.
So the saving txt method have put in animation function
'''
class heteregeneousModel:
    def __init__(self,num_people,parameter,history=False,processHistory=False,Distance="gradient",method="uniform",
    testProbabilityMode=False):
        """
          Initialize the representation of the state vector as a list of integers, 
         [1, 0, 0, ...]
         And simulate the locations data for each object as np array([num_people,2])
        """
        self.state = np.zeros((num_people))
        self.state[0] = 1
        self.num_people=num_people
        self.parameter=parameter
        self.processHistory=processHistory
        self.history=history
        #self.beta0=parameter[0] #baseline beta
        #self.phi=parameter[1]     #scale parameter
        self.gamma=parameter[-1] #remove rate
        self.Distance=Distance
        self.method=method
        #change start with here
        #I need add the geo-information into state
        self.geo=cr.geodata(num_people,method,xbound=100.0,ybound=100.0,history=history)
        #cr.plotcoor(self.geo)
        self.DistanceMatrix=cr.DistanceMatrix(self.geo)
        self.BetaMatrix=cr.BetaMatrix(self.DistanceMatrix,np.delete(parameter,-1),Distance)
        #gp.BetaMatrixPlot(self.DistanceMatrix,self.BetaMatrix)
        self.mainProcess(processHistory,testProbabilityMode)
    def transform_prob(self,state):
        """
        transform the current state to next step:
        suspect(0)->infectious(1) with probability Lambda(state,beta)
        infectious(1)->removal(2) with probability np.exp(-gamma)
        returns the state after 1 day
        """
        p_S_to_I = self.Lambda(state)
        #now it is a num_people array
        p_I_to_R = 1-np.exp(-self.gamma)
        prob_transitions = np.zeros((self.num_people,))
        prob_transitions = p_S_to_I
        prob_transitions[state==1] = p_I_to_R
        prob_transitions[state==2] = 0
        return prob_transitions
    
    def transformation(self,state,prob_transitions):
        
        Binomi=np.random.uniform(0, 1, prob_transitions.shape)
        transitions = prob_transitions > Binomi
        #
        #self.InfectedOutput(state,prob_transitions,Binomi)
        #Debug function
        return state + transitions
    def transform(self,state,testProbabilityMode):
        prob_transitions=self.transform_prob(state)
        if testProbabilityMode==True:
            print("Transform probability is")
            print(prob_transitions)
        return self.transformation(state,prob_transitions)
    def InfectiousStop(self,state):
        """
         When all infectious are remove, then we can stop the 
         """
        if (np.sum(state==1)==0):
            return False                           #all infected people have removed
        else:
            return True
    def mainProcess(self,processHistory=False,testProbabilityMode=False):                    #simulation procedure 
        """
         This is the process for the infection infect population in risk
         For each day, use transform function to get the state for next day
         return the number of infected people everyday
    
        """
        stateInit=self.state
        record = [stateInit.copy()]
        state = stateInit                            #state is the current state
        """
        for i in range(0, num_days):                 
            state=transform(state)
        record.append(state.copy())              #record the everday state
        for auto stop, use while instead of for
         """
        nday=0
        if processHistory==True:
            try:
                self.record=np.loadtxt("record"+self.Distance+".txt")
                cout1 = [np.sum(day==1) for day in record] # list comprehension
                cout2=np.sum(record[-1]==2)# the total people who are infected(finally all of them are removed)
                self.cout1=cout1 
                self.cout2=cout2
                [self.nday,numpeople]=self.record.shape
                return [cout1,cout2]
            except FileNotFoundError:
                print("no file found, auto create new file") 
        #########################################################
        #core code
        while (self.InfectiousStop(state)):
            state=self.transform(state,testProbabilityMode)
            record.append(state.copy())
            nday+=1
        ##########################################################
        cout1 = [np.sum(day==1) for day in record] # list comprehension
        cout2=np.sum(record[-1]==2)# the total people who are infected(finally all of them are removed)
        self.record=np.array(record)
        self.cout1=cout1#count1 is the daily record of the infected people
        self.cout2=cout2#count2 is the final infected size of the infected people
        self.nday=nday
       
        return [cout1,cout2]
   
    def Lambda(self,state):
        '''
        Lambda function returns a transform probability for every state to next step(S->I, I->R)
        This Lambda function is based on the "infected pressure" model which is differ from the previous one "Infected probability model".
        Infected pressure model is talking about that each infected people have a pressure to the surrounding suspected people, instead of
        the "probability at least one Infected object infect object j". The probability of j get infected in the end of the day is the 
        cumulative pressure for the infected objects surrouding j.
        For more detail, plz check "record 6.23.pdf" 2.1
        '''
        probInfect=np.zeros((self.num_people))
        '''
        for i in zip(*np.where(state==0)):
            probInfect[i]=sum(self.BetaMatrix[i][state==1])
        '''
        ################################################
        probInfect[state==0] = self.BetaMatrix[:, state==1].sum(1)[state==0]
        probInfect=1-np.exp(-probInfect)
        #################################################
        return probInfect
    def Animate(self):
        '''
        Function for show the animation of infectious
        '''
        numframes = self.nday+1
        numpoints = self.num_people
        color_data =np.zeros((numframes, numpoints)) 
        color_data[self.record==1]=0.9
        color_data[self.record==2]=0.5
        x, y, c = np.random.random((3, numpoints))
        x=self.geo[:,0]
        y=self.geo[:,1]
        c=np.zeros(numpoints)
        c[0]=1
        fig = plt.figure()
        scat = plt.scatter(x, y, c=c, s=70)
        def update_plot(i, data, scat):
            scat.set_array(data[i])
            return scat,
        ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat),interval=1000)
        #Writer = animation.writers[]
        #ani.save('animation.mp4',fps=30,writer=Writer)
        ani.save('the_movie.mp4', writer = 'ffmpeg', fps=1)
        plt.show()
        if self.processHistory==False:
            print("Save this epdemic?y/n")
            if input()=="y":
                np.savetxt("record"+self.Distance+".txt",self.record)
            else:
                pass
    def InfectedOutput(self,state,prob_transitions,Binomi):
        '''
        function for debugging
        Print the dice and the infected probability for state changed individuals
        '''
        print("Probabitliy")
        print (prob_transitions[prob_transitions>Binomi])
        print("dice")
        print(Binomi[prob_transitions>Binomi])
        print("Location")
        print (self.geo[prob_transitions>Binomi,:])
    def savecsv(self,filename="uni"):
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.savetxt("epidemicsResult"+filename+timestr+".csv", self.record, delimiter=",")
        np.savetxt("AddressResult"+filename+timestr+".csv", self.geo, delimiter=",")
    
#model1=heteregeneousModel(20,0.15,0.03,10,True)
"""
print(model1.record)
print(model1.cout1)
print(model1.cout2)
plt.scatter(model1.geo[:,0],model1.geo[:,1])
plt.show()
record=np.array(model1.record)
print(record[0,:])
plt.scatter(model1.geo[record[0,:]==1,0],model1.geo[record[0,:]==1,1],color="red")
plt.scatter(model1.geo[record[0,:]==0,0],model1.geo[record[0,:]==0,1],color="blue")
plt.show()
#model1.Animate()
print(model1.record)
"""

'''
def main():
    numframes = model1.nday+1
    numpoints = model1.num_people
    color_data =np.zeros((numframes, numpoints)) 
    color_data[model1.record==1]=0.9
    color_data[model1.record==2]=0.5
    x, y, c = np.random.random((3, numpoints))
    x=model1.geo[:,0]
    y=model1.geo[:,1]
    c=np.zeros(numpoints)
    c[0]=1
    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=70)
    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat),interval=1000)
    plt.show()

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,
'''
#model1.Animate()
