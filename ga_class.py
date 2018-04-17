# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:57:41 2017
@author: Rohit Annigeri
@author: Ali Marjaninejad
"""

import random
import numpy as np
from deap import tools
from deap import base, creator

"""
class " GenticAlgorithm "
Notes: Creates a base object to run a genetic algorithm to minimize a
in built cost function
"""
class GenticAlgorithm(object):
    """
    function " __init__ "
    Notes: initializes gentic algorithm specific parameters
    individual size, population size, constraints, tolerance
    max number of generations, top k (BEST) individuals to keep
    number of children from each couple
    """
    def __init__(self,param=None):
        # dimensionality of a single population member
        self.__NUM_INDIVIDUALS = 2
        # population per generation allowed in the GA
        self.__TOTAL_POP = 32
        #total children created per breeding = SIBLINGS_PER_COUPLE*2
        self.__SIBLINGS_PER_COUPLE = 1 
        # preserve BEST number of members per generation
        self.__BEST = 6
        # error tolerance value
        self.__TOLERANCE = self.__fit_scaling(0.0087)[0]
        # LOWER bound values
        self.__LOW = [-4,-7]
        # Upper bound values
        self.__HIGH = [0,0]
        # total number of generations allowed to find solution
        self.__TOTAL_GENERATIONS = 5000
        #default R matrix
        self.__R = np.linalg.inv(np.array([[-2,0],[0,-1.5]]))
        
        self.__mean_error = []
        self.__min_error = []
        self.__seed_pop = []
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.__toolbox = base.Toolbox()
        
        self.__toolbox.register("attribute_0", random.uniform, self.__LOW[0], self.__HIGH[0])
        self.__toolbox.register("attribute_1", random.uniform, self.__LOW[1], self.__HIGH[1])
        self.__toolbox.register("individual", tools.initCycle, creator.Individual,
                         (self.__toolbox.attribute_0,self.__toolbox.attribute_1), n=1)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)
        self.__toolbox.register("mate", self.__mate)
        self.__toolbox.register("mutate",self.__mutate,k=3)
        self.__toolbox.register("select", tools.selBest, k=2)
        self.__toolbox.register("evaluate", self.__evaluate)
        self.__toolbox.register("add_new_ind",self.__add_new_ind)

    """
    function " constrained_individual "
    Notes: creates a random individual within constraints
    """    
    def constrained_individual(self):
        individual = []
        for i in range(self.__NUM_INDIVIDUALS):
            individual.append(random.uniform(self.__LOW[i],self.__HIGH[i]))
        return individual
    """
    function " set_R "
    Notes: Sets the R matrix for cost function evaluation
    """    
    def set_R(self, R):
        if(R.shape != (2,self.__NUM_INDIVIDUALS)):
            print('Dimension needs to be (2,{})'.format(self.__NUM_INDIVIDUALS))
            return
        self.__R = R
    """
    function " __evaluate "
    Notes: cost function 
    """    
    def __evaluate(self,individual,true_angles):
        pred_angles = np.dot(self.__R,np.array(individual))
        loss = np.linalg.norm(true_angles-pred_angles)
        return loss
    """
    function " __scaling "
    Notes: scale the individuals within range of 0-255, to perform efficient 
    bitwise crossover and vice versa
    """        
    def __scaling(self,individual,direction='shrink'):
        for i in range(len(individual)):
            if direction == 'shrink':
                scaled_val = int((individual[i]-self.__LOW[i])*255//(self.__HIGH[i]-self.__LOW[i]))
            else:
                scaled_val = (individual[i]/255)*(self.__HIGH[i]-self.__LOW[i]) + self.__LOW[i]
            individual[i] = scaled_val
        return 
    """
    function " __bit_crossover "
    Notes: bitwise crossover of two scaled individuals
    """       
    def __bit_crossover(self,ind1,ind2):
        sibling1 = self.__toolbox.clone(ind1)
        sibling2 = self.__toolbox.clone(ind2)
        
        self.__scaling(sibling1,direction='shrink')
        self.__scaling(sibling2,direction='shrink')
        
        swap_index = random.randint(1,7)
        mask_1 = (0x01 << swap_index) - 1
        mask_2 = 0xFF - mask_1
        for i in range(len(ind1)):
            sibling1[i] = (sibling1[i] & mask_1) | (sibling2[i] & mask_2)
            sibling2[i] = (sibling1[i] & mask_2) | (sibling2[i] & mask_1)
        self.__scaling(sibling1,direction='restore')
        self.__scaling(sibling2,direction='restore')
        
        return sibling1, sibling2
        """
    function " __bit_crossover "
    Notes: bitwise crossover of two scaled individuals
    """       
    def __averaging(self,ind1,ind2):
        sibling1 = self.__toolbox.clone(ind1)
        sibling2 = self.__toolbox.clone(ind2)
        
        self.__scaling(sibling1,direction='shrink')
        self.__scaling(sibling2,direction='shrink')
        
        for i in range(len(ind1)):
            sibling1[i] = (sibling1[i] + sibling2[i])/2
        self.__scaling(sibling1,direction='restore')
        self.__scaling(sibling2,direction='restore')
        
        return sibling1
    """
    function " __mate "
    Notes: bitwise crossover for selected set of individuals
    creates 2*siblings_per_couple children
    """         
    def __mate(self,population):
        children = []
        for ind1, ind2 in zip(population[::1],population[1::1]):
            for i in range(self.__SIBLINGS_PER_COUPLE):
                child1, child2 = self.__bit_crossover(ind1, ind2)    
                children.append(child1)
                children.append(child2)
#            child = self.__averaging(ind1,ind2)
#            children.append(child)
        for child in children:
            del child.fitness.values
        return population + children
    """
    function " __add_new_ind "
    Notes: add new random individuals from outside the current population
    """          
    def __add_new_ind(self,population):
        new_ind = self.__toolbox.population(n=self.__TOTAL_POP-len(population))
        for ind in new_ind:
            del ind.fitness.values
        return population+new_ind
    """
    function " __mutate "
    Notes: add random bit mutations to top individuals through rescaling
    """              
    def __mutate(self,population,k):
        mutations = self.__toolbox.clone(population[:k])
        for ind in mutations:
            self.__scaling(ind,direction='shrink')
            for i in range(len(ind)):
                rand_element = random.randint(0,255)
                ind[i] = ind[i]^rand_element
            self.__scaling(ind,direction='expand')
            del ind.fitness.values
        return population+mutations
    """
    function " __fit_scaling "
    Notes: rescaling fitness values using sigmoid function
    """ 
    def __fit_scaling(self,fitness):
        return fitness,0
#        return  (1/(1 + np.exp(-fitness))),0
    """
    function " get_fit_error "
    Notes: returns mean and min fitness error across all generations
    """ 
    def get_fit_error(self):
        return self.__mean_error, self.__min_error
    
    """
    function " run "
    Notes: run the GA for a set of given targets and use a given best
    solution of previous generation
    """    
    def run(self,true_angles=[np.pi, 2*np.pi],prev_best = None):
        self.__mean_error = []
        self.__min_error = []
        top_candidate = []
        
        pop = self.__toolbox.population(n=self.__TOTAL_POP)
        
        if (prev_best is not None):
            for i in range(len(pop[0])):
                pop[0][i] = prev_best[i]
                
        if len(self.__seed_pop) != 0 and prev_best is not None:
            pop[:len(self.__seed_pop)] = list(map(self.__toolbox.clone,self.__seed_pop))
            
        fitnesses = [self.__toolbox.evaluate(ind,true_angles) for ind in pop]
        for ind, fit_value in zip(pop,fitnesses):
            ind.fitness.values = self.__fit_scaling(fit_value)
            
        n_gen = 0
        while n_gen < self.__TOTAL_GENERATIONS:
            n_gen += 1
            offspring = self.__toolbox.select(pop,k=self.__BEST)
            offspring = list(map(self.__toolbox.clone,offspring))
            #breeding of top BEST members
            offspring = self.__toolbox.mate(offspring)
            #add random mutated top BEST members
            offspring = self.__toolbox.mutate(offspring,k=self.__BEST)
            # add outside (random) members
            offspring = self.__toolbox.add_new_ind(offspring)
            #update fitness values for new members
            for ind in offspring:
                if ind.fitness.valid == False:
                   raw_fit = self.__toolbox.evaluate(ind,true_angles)
                   ind.fitness.values = self.__fit_scaling(raw_fit)
                               
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
            self.__mean_error.append(np.mean(fits))
            
            #test best candidate for early exit
            top_candidate = self.__toolbox.select(pop,k=1)[0]
            min_fit = top_candidate.fitness.values[0]
            self.__min_error.append(min_fit)
            
            if min_fit <= self.__TOLERANCE:
                self.__seed_pop = self.__toolbox.select(pop,k=self.__BEST)
                break
            
        return top_candidate