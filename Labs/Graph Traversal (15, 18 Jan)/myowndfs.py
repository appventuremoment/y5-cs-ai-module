def dfs(graph, start):
    frontier = [start]
    visited = []

    while frontier:
        currnode = frontier.pop()
        visited.append(currnode)
        for child in graph[currnode]:
            if child not in visited and child not in frontier:
                frontier.append(child)

    return visited



graph={
    'A':['C','E'],
    'B':[],
    'C':['B','G'],
    'D':[],
    'E':['H'],
    'H':['D'],
    'G':[]
}
dfs(graph,'A')
print(dfs(graph, 'A'))

#https://medium.com/nerd-for-tech/graph-traversal-in-python-depth-first-search-dfs-ce791f48af5b