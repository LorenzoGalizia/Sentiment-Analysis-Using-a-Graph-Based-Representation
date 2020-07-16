import numpy as np

def degree_centrality(graph):
    A = graph.degree()
    return(A)

def closeness_centrality(graph):
    close = {}
    nodes = list(graph.vs())
    for n in nodes:
        sp = np.ma.masked_invalid(graph.shortest_paths_dijkstra(source = n, target = None, weights="weight")[0])
        close[n] = (len(nodes) - 1.0)/sp.sum()
    return close


def betweeness_centrality(graph):
    btw = graph.betweenness()
    return(btw)
    
    
def ev_centrality(graph):
    ev = graph.evcent()
    return(ev)
