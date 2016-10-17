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
model1=gc2.heteregeneousModel(population,[0.4,0.1,1.5,0.3],True,True,"gradient")
#model1.Animate()
estimate=lk2.Estimation(model1.record,model1.geo)
InitialGP=np.zeros(population*(population-1)/2)#zero GP and no transform
Metro=mp3.multiMetropolis(4000,[partial(estimate.GammaPriorGeneralPosterior,i=0),
partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2)],
[0.1,0.1,0.1],[0.4,0.3,0.4],InitialGP)
Metro.showplot(0)
Metro.showplot(1)
Metro.showplot(2)
Metro.plotcountour(0,1)
Metro.printall(0)
Metro.printall(1)
Metro.printall(2)
Metro.saveResult("NoGP","POP50Para")