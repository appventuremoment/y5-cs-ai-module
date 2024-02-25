import numpy as np
import random as rnd

distance = [[0.00, 28.02, 17.12, 27.46, 46.07],
            [28.02, 0.00, 34.00, 25.55, 25.55],
            [17.12, 34.00, 0.00, 18.03, 57.38],
            [27.46, 25.55, 18.03, 0.00, 51.11],
            [46.07, 25.55, 57.38, 51.11, 0.00]]

decision_variables = [0, 1, 2, 3, 4]

#Exercise 1
def initialize():
    pop_bag = []

    for i in range(10):
        pop_bag.append(rnd.sample(decision_variables, 5))
    return np.array(pop_bag)

#Exercise 2
def fitness_function(solution):
    tot_distance = 0
    for x in range(4):
        tot_distance += distance[solution[x]][solution[x + 1]]
    
    return tot_distance


def eval_fit_population(pop_bag):
    result = {}
    fit_vals_1st = []
    solutions = []
    for solution in pop_bag:
        fit_vals_1st.append(fitness_function(solution))
        solutions.append(solution)
    result['fit_vals'] = fit_vals_1st
    min_wgh = [np.max(list(result["fit_vals"])) - i for i in list(result["fit_vals"])]
    # print(min_wgh)
    # print(sum(min_wgh))
    result["fit_wgh"] = [i / sum(min_wgh) for i in min_wgh]
    result["solution"] = np.array(solutions)
    return result


#Exercise 3
def pickOne(pop_bag):
    fit_bag_evals = eval_fit_population(pop_bag)
    choice = fit_bag_evals['fit_wgh'].index(rnd.choice(fit_bag_evals['fit_wgh']))
    return fit_bag_evals['solution'][choice]

#Exercise 4
def crossover(solA, solB):
    # print('cross_over,pA', pA)
    # print('cross_over,pB', pB)
    child = [np.nan for i in range(len(solA))]
    half = rnd.randint(1, 2)
    if half == 1:
        child[:2] = solA[:2]
        otherhalf = [x for x in solB if x not in child]
        child[2:] = otherhalf
    else:
        child[2:] = solA[2:]
        otherhalf = [x for x in solB if x not in child]
        child[:2] = otherhalf
    return child


#Exercise 5
def mutation(sol):
    result = sol.copy()
    firstindex = rnd.randint(0, 4)
    secondindex = rnd.randint(0, 4)
    result[firstindex], result[secondindex] = result[secondindex], result[firstindex]
   
    return result


# Create the initial population bag
#Test case: uncomment after you have implemented exercise 1 
pop_bag = initialize()
print(pop_bag)

# test fitness function
for chromosome in pop_bag:
    print('chromosome',chromosome,',fitness:',fitness_function(chromosome))

print(eval_fit_population(pop_bag))
print(pickOne(pop_bag))
print(crossover([1, 4, 2, 3, 0],  [2, 0, 4, 3, 1]))


# Iterate over all generations
for g in range(200):
    # Calculate the fitness of elements in population bag
    pop_bag_fit = eval_fit_population(pop_bag)
    print(pop_bag_fit)
    # Best individual so far
    best_fit = np.min(pop_bag_fit["fit_vals"])
    best_fit_index = pop_bag_fit["fit_vals"].index(best_fit)
    best_solution = pop_bag_fit["solution"][best_fit_index]

    # Check if we have a new best
    if g == 0:
        best_fit_global = best_fit
        best_solution_global = best_solution
    else:
        if best_fit <= best_fit_global:
            best_fit_global = best_fit
            best_solution_global = best_solution

    # Create the new population bag
    new_pop_bag = []
    for i in range(10):
        # Pick 2 parents from the bag
        pA = pickOne(pop_bag)
        pB = pickOne(pop_bag)
        print('pA', pA)
        print('pB', pB)
        new_element = pA
        # Crossover the parents
        if rnd.random() <= 0.87:
            new_element = crossover(pA, pB)
        # Mutate the child
        if rnd.random() <= 0.7:
            new_element = mutation(new_element)
        # Append the child to the bag
        new_pop_bag.append(new_element)
    # Set the new bag as the population bag
    pop_bag = np.array(new_pop_bag)
    print('generation:', g)
    print('best fit global', best_fit_global)
    print('best soln', best_solution_global)
