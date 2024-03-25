#set page(columns: 3)
#set text(size:9pt)
#show heading.where(
  level: 3
): it => text(
  size: 11pt,
  weight: "bold",
  it.body
)
#set page(margin: (
  top: 0.5cm,
  bottom: 0.5cm,
  x: 0.5cm
))

#let achievement(name) = [#text(
  name, 
  // style: "oblique",
  weight: "bold",
  size:8pt,
)]

= Chapter 2
- Unsupervised learning is unlabelled data in order to find patterns
- Reinforcement learning is using unlabelled data and trial of error, then improve its performance from the results of each trial
- No Free Lunch Theorem: There is no one size fits all model

==== Bias Variance Tradeoff

Bias systematically incorrect for certain values. Variance is fluctuation in output when different inputs, essentially the flexibility.
- Balance of precision and recall, F1 Score = 2 \* (Precision \* Recall) / (Precision + Recall)
- AUC-ROC = Area under curve of TP rate against FP rate
#image("image.png", width: 100%)

==== To solve over/underfitting
- Choose a simpler model
- Regularisation is limiting some of the parameter coefficients close to 0
- Reduce parameters of model
- Gather more data
- Reduce outliers and errors in training data
- Solving underfitting is above inverse plus adding better features to training data

==== SVM
Supervised BINARY classification, for each dimension (both axis are independant variables), a best hyperplane line is drawn which maximises the distance to the closest distance in each class

==== Linear Regresion
#image("image2.png", width: 70%)

==== Logisitic Regression Functions
Sigmoid, Logistic and Logit/Log-odds function
#image("image3.png", width: 85%)
#image("image4.png", width: 65%)
#image("image5.png", width: 85%)

==== Distance Functions

where x and y are vectors with k - 1 dimensions
#image("image6.png", width: 75%)

==== Perceptron Neuron
Refer to below for how to train it, except using this change in weight:
#image("image11.png", width: 105%)

==== Backpropagation 
It uses a sigmoid function so it never reaches 0 or 1
#image("image16.png", width: 105%)

To optimise backpropagation to be faster, we can:
  1. Use the previous change in weight of the neuron \* some constant x in the change in weight of neuron formula such that it avoids local minima
  2. Use hyperbolic tangent function 2a/(1+e^-bx) - a where a,b are constants
  3. Increase the learning rate

ReLU Function is 0 below 0 and linearly increasing above 0

==== Pooling layers
#image("image15.png", width: 65%)

== Chapter 3
==== BFS/DFS 
BFS will always terminate with the best solution but is computationally expensive and chronically long. DFS is reverse (might get stuck in infinite loop)

#set text(size:7pt)
```py 
def bfs(graph, start):
    frontier = [start]
    visited = []

    while frontier:
        # replace 0 with nothing for dfs
        currnode = frontier.pop(0) 
        visited.append(currnode)
        for child in graph[currnode]:
            if child not in visited and child not in frontier:
                frontier.append(child)

    return visited
```
#set text(size:8pt)
==== A\*
For A\*, the estimated cost for any node is the shortest path cost from the start node to the node plus the estimated distance (using the distance functions) to the goal node.

#image("image20.png", width: 105%)

== Chapter 4 
==== GA Lifecycle
#image("image21.png", width: 65%)

==== Selection Methods
- Roulette Wheel: Each chromosome has a weighted chance to be chosen as a parent based on its fitness as compared to the fitness sum of the population. Select one parent based on the chances
- Stochastic Universal Sampling: Roulette Wheel but for choose parents at the same time
- K-way Tournament: Choose K random chromosomes from the population and out of those chosen, pick the best one to be parent
- Rank: Only for population with chromosomes with close fitness, the chromosomes are sorted by rank and uses the rank as the weighted chance to be selected as parent

==== Crossover Methods
- N-point crossover: Chooses N points to split both parents and alternate each section as the child
- Uniform crossover: Every allele of each parent has a chance (weighted or not) to be chosen as offspring allele

==== Mutation Operators
- Bitflip: Not.
- Swap: Swap allele of 2 random positions in child
- Scramble: Subset of genes are scarambled
- Inversion: Subset of genes are reversed in order

==== Selection Operators (Steady state populations only)
- Age based selection: Oldest is removed to be replaced by new offspring
- Fitness based selection: I am sure you can figure this one out

