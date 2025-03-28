def bfs(graph,node):
    
    # node is the starting position
    # graph is the graph in dictionary format
    visited=[]
    queue=[]
    out = []
    
    queue.append(node)
    visited.append(node)
    
    while queue:
        s=queue.pop()
        out.append(s)
        for x in graph[s][::-1]:
            if x not in visited:
                visited.append(x)
                queue.append(x)
            
    print(out)
    

graph={
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
    'E':['F'],
    'F':[]
}
bfs(graph,'A')
# this will return the sequence of A,B,D,E,F,C

#https://medium.com/nerd-for-tech/graph-traversal-in-python-depth-first-search-dfs-ce791f48af5b
