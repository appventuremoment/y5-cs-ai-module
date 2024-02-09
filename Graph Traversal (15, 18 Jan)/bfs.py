def bfs(graph,node):
    
    # node is the starting position
    # graph is the graph in dictionary format
    visited=[]
    queue=[]
    
    queue.append(node)
    visited.append(node)
    
    while queue:
        s=queue.pop()
        print(s)
        for x in graph[s][::-1]:
            if x not in visited:
                visited.append(x)
                queue.append(x)

    return visited

graph={
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
    'E':['F'],
    'F':[]
}
print(bfs(graph, 'A'))

#im pretty sure this is bfs