import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend
import random
import math

random.seed(42)

city_count = 50
ant_count = city_count*2
iterations = 50
stepsize = 1
evaporation = 0.7
alpha = 1.
beta = 4
Q = 1

patience=iterations
patience_counter = 0
shortest_path_length = None
shortest_path = []
# size = city_count
size = int(math.sqrt(city_count))

def generate_weighted_graph(num_nodes, size=10):
    G = nx.complete_graph(num_nodes)

    # Generate random positions for nodes
    pos = {node: (random.random()*size, random.random()*size) for node in G.nodes()}

    # Calculate Euclidean distances and assign weights
    for edge in G.edges():
        node1, node2 = edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        G.edges[edge]['weight'] = distance

    return G, pos


def calc_probs(G, l):
    for i in range(l):
        city_denominator = 0
        for j in range(l):
            if i == j:
                continue
            city_denominator += G.edges[(i,j)]['tau']**alpha * (1/G.edges[(i,j)]['weight'])**beta
        for j in range(l):
            if i == j:
                continue
            G.edges[(i,j)]["trans_prob"] = (G.edges[(i,j)]['tau']**alpha * (1/G.edges[(i,j)]['weight'])**beta) / city_denominator


def move_ant(G, ant):
    # get list of open candidates
    open = set(G.nodes) - set(ant["tabu"])
    open_list = list(open)
    pos = ant["pos"]

    if 0 == len(open):
        # last iteration... go home and close loop
        target = ant["home"]
    else:
        city_denominator = 0 
        for j in open_list:
            city_denominator += G.edges[(pos,j)]['tau']**alpha * (1/G.edges[(pos,j)]['weight'])**beta
        probs = []
        for j in open_list:
            probs.append((G.edges[(pos,j)]['tau']**alpha * (1/G.edges[(pos,j)]['weight'])**beta) / city_denominator)
        probs = list(np.array(probs) / np.sum(probs))
        
        target = np.random.choice(list(open), p=probs)
    
    # add target to trail
    ant["trail"].append(target)
    
    # add target to tabu
    ant["tabu"].add(target)
    
    # change pos to target
    ant["pos"] = target
    
    # add edge weight to length
    ant["length"] += G.edges[(pos,target)]["weight"]
        

def show_graph(G, node_positions, iter, red_path=[]):
    
    edge_intensities = [G.edges[edge]['tau'] for edge in G.edges]
    edge_intensities = list((np.array(edge_intensities)-np.min(edge_intensities))/np.max(edge_intensities))
    # Draw the graph with pheromone concentration
    axes[0].clear()
    axes[0].set_title(f"Normalized Pheromone Intensity at iter {iter}")
    nx.draw(G, pos=node_positions,
            with_labels=True,
            width=[i*10 for i in edge_intensities],
            edge_color=[1 for i in edge_intensities],
            edge_cmap=plt.cm.Blues,
            edge_vmin=0.5,
            edge_vmax=1.0,
            node_size=200,
            ax=axes[0]
            )
    edge_probs = np.array([G.edges[edge]["trans_prob"] for edge in G.edges])
    edge_probs = list((np.array(edge_probs)-np.min(edge_probs))/np.max(edge_probs))
    axes[1].clear()
    axes[1].set_title(f"Normalized Transition Probability at iter {iter}")
    nx.draw(G, pos=node_positions,
            with_labels=True,
            width=[i*7 for i in edge_probs],
            edge_color=[1 for i in edge_probs],
            edge_cmap=plt.cm.Blues,
            edge_vmin=0.0,
            edge_vmax=1.0,
            node_size=200,
            ax=axes[1]
            )
    
    if len(red_path):
        edgelist = [(red_path[i],red_path[i+1]) for i in range(len(red_path)-1)]
        nx.draw_networkx_edges(G, node_positions, edgelist=edgelist, edge_color="r", ax=axes[0])
        nx.draw_networkx_edges(G, node_positions, edgelist=edgelist, edge_color="r", ax=axes[1])
    

        
def do_aco_iter(frame):
    global shortest_path_length
    global shortest_path
    nodes = G.nodes
    for it in range(stepsize):
        # place ants
        ants = []
        for i in range(ant_count):
            node = i%len(nodes)
            ants.append({
                "home": node,
                "pos" : node ,
                "tabu" : set([node,]),
                "trail": [node,],
                "length": 0 
                }) # equally distribute ant home positions over all towns
        for i in range(len(nodes)): # until every ant visited every town
            for a in range(len(ants)): # for each ant: visit next town
                ant = ants[a]
                move_ant(G, ant)
                
        # calculate delta tau values
        for a in range(len(ants)):
            trail = ants[a]["trail"]
            for i in range(len(trail)-1):
                edge = tuple(sorted(trail[i:i+2]))
                G.edges[edge]["delta_tau"] += Q/ants[a]["length"]
        
        # calculate new tau values
        for i in range(len(nodes)):
            for j in range((len(nodes))):
                if i == j:
                    continue
                edge = (i,j)
                G.edges[edge]['tau'] = G.edges[edge]['tau'] * evaporation + G.edges[edge]['delta_tau'] 

        # calculate new probability values
        calc_probs(G, len(nodes))

        # memorize shortest path
        if None == shortest_path_length:
            shortest_path_length = ants[0]["length"]
        for a in range(len(ants)):
            l = ants[a]["length"]
            if shortest_path_length > l:
                shortest_path_length = l
                shortest_path = ants[a]["trail"]
                print(f"Iter:{frame*stepsize+1+it}\tFound new shortest path with length {shortest_path_length}: {shortest_path[:20]}")

            # reset ant
            ants[a]["length"] = 0
            ants[a]["trail"] = []
            ants[a]["tabu"] = set([ants[a]["home"],])
            ants[a]["pos"] = ants[a]["home"]
    show_graph(G, node_positions, frame*stepsize+1+it, shortest_path)




if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    G, node_positions = generate_weighted_graph(city_count, size=size)
    random.seed()
    nx_solution = nx.approximation.traveling_salesman_problem(G) # christofides algorithmus
    edges = [(nx_solution[i],nx_solution[i+1]) for i in range(len(nx_solution)-1)]
    nx_length = np.sum([G.edges[e]["weight"] for e in edges])
    print(f"Networkx found path with length {nx_length}: {nx_solution[:20]}")
    nodes = G.nodes

    # set init tau values
    tau = np.ones((len(nodes), len(nodes))) * 0.1
    for edge in G.edges:
        G.edges[edge]['tau'] = 0.1
        G.edges[edge]['delta_tau'] = 0
    calc_probs(G, len(nodes))
    # Create an animation
    ani = FuncAnimation(fig, do_aco_iter, frames=int(iterations/stepsize), repeat=False)
    plt.show()
    


