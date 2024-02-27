import numpy as np
import random as rnd

product_ratings = {
    'chocolate': [0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.7, 0.8, 0.9, 0.8, 0.7],
    'biscuit': [0.7, 0.8, 0.6, 0.5, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.6, 0.7, 0.8, 0.7, 0.6],
    'ice_cream': [0.9, 0.8, 0.9, 0.7, 0.8, 0.7, 0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.8],
    'coffee': [0.8, 0.7, 0.8, 0.9, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.8],
    'fruit_juice': [0.7, 0.6, 0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5, 0.7, 0.8, 0.6, 0.7, 0.8],
    'energy_drink': [0.6, 0.7, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6, 0.7, 0.8],
    'snack_bar': [0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.7, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6],
    'yogurt': [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
    'soda': [0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6],
    'water': [0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
    'cereal': [0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7],
    'bread': [0.7, 0.6, 0.7, 0.8, 0.7, 0.6, 0.8, 0.9, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.8],
    'milk': [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7],
    'cheese': [0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.8],
    'butter': [0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7, 0.6, 0.8, 0.7, 0.8, 0.7, 0.6, 0.7, 0.8]
}


decision_variables = [x for x in range(15)]

def initialize():
    pop_bag = []
    for i in range(20):
        pop_bag.append([rnd.randint(0, len(decision_variables) - 1) for _ in range(15)])
    return np.array(pop_bag)


def fitness_function(solution):
    indices = list(product_ratings.keys())
    market = 0
    fitness = 0
    for ind in solution:
        fitness += product_ratings[indices[ind]][market]
        market += 1
    return fitness

def kTournament(pop_bag):
    indices = []
    while len(set(indices)) != 3:
        indices = np.random.choice(len(pop_bag), 3)

    contenders = [fitness_function(pop_bag[x]) for x in indices]
    finalindex = contenders.index(max(contenders))
    return pop_bag[indices[finalindex]]


def eval_fit_population(pop_bag):
    result = {}
    fit_vals_1st = []
    solutions = []
    for solution in pop_bag:
        fit_vals_1st.append(fitness_function(solution))
        solutions.append(solution)
    result['fit_vals'] = fit_vals_1st
    min_wgh = [np.max(list(result["fit_vals"])) - i for i in list(result["fit_vals"])]
    result["fit_wgh"] = [i / sum(min_wgh) for i in min_wgh]
    result["solution"] = np.array(solutions)
    return result

def StochasticUniversalSampling(pop_bag):
    fit_bag_evals = eval_fit_population(pop_bag)
    choice = np.random.choice(20, 40, p=fit_bag_evals['fit_wgh'])
    parents = []
    for picked in choice:
        parents.append(fit_bag_evals['solution'][picked])
    return parents

def crossover(solA, solB):
    n = len(solA)
    cutOff1, cutOff2 = np.sort(np.random.choice(n, 2))
    child = solB
    child[cutOff1:cutOff2] = solA[cutOff1:cutOff2]
    return child


def swapmutation(sol):
    result = sol.copy()
    firstindex = rnd.randint(0, 14)
    secondindex = rnd.randint(0, 14)
    result[firstindex], result[secondindex] = result[secondindex], result[firstindex]
   
    return result

print(swapmutation([x for x in range(15)]))

# Create the initial population bag
pop_bag = initialize()
print(pop_bag)


# Iterate over all generations
for g in range(200):
    # Calculate the fitness of elements in population bag
    pop_bag_fit = eval_fit_population(pop_bag)
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

    #Elitism
    new_pop_bag.append(list(pop_bag).pop(best_fit_index))


    for i in range(19):
        # Pick 2 parents from the bag
        pA = kTournament(pop_bag)
        pB = kTournament(pop_bag)
        print('pA', pA)
        print('pB', pB)
        new_element = pA
        # Crossover the parents
        if rnd.random() <= 0.87:
            new_element = crossover(pA, pB)
        # Mutate the child
        if rnd.random() <= 0.7:
            new_element = swapmutation(new_element)
        # Append the child to the bag
        new_pop_bag.append(new_element)

    # Set the new bag as the population bag
    pop_bag = np.array(new_pop_bag)
    print('generation:', g)
    print('best fit global', best_fit_global)
    print('best soln', best_solution_global)