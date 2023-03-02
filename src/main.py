# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cdlib.algorithms

import GA
import numpy as np
import networkx as nx
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn.preprocessing import MinMaxScaler
from cdlib import evaluation, algorithms, NodeClustering, ensemble
import math
import multiprocessing as mp
import time 


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def readParameter():
    print("Begin!!!!!!!!")
    path = "../data/Parameters/"
    for file in os.listdir(path):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            # get the Network File
            file_path = f"{file}"
            print("Begin: " + file_path)
            iterationSize = []
            populationSize = []
            crossoverPc = []
            mutationPm = []
            # read net file and get parameters
            with open(path + file_path) as f:
                while True:
                    line = f.readline().strip()
                    # if line is empty
                    # end of file is reached
                    if not line:
                        break
                    if not line.__contains__("N") and not line.__contains__(
                            "Pc") and not line.__contains__("Pm"):
                        parameters = line.split('\t')
                        # parameterN = float(parameters[0])
                        iterationSize.append(int(float(parameters[0])))
                        populationSize.append(int(float(parameters[1])))
                        crossoverPc.append(float(parameters[2]))
                        mutationPm.append(float(parameters[3]))

            return iterationSize, populationSize, crossoverPc, mutationPm
        
def task(Gi, N, I, Pc, Pm):
    return GA.ga_community_detection(Gi, N, I, 1.5, Pc, Pm)

def wrapper(Gi, N, I, h, Pc, Pm, parametrization):

    return (GA.ga_community_detection(Gi, N, I, 1.5, Pc, Pm), parametrization)

def run(Gi, parameters, file_path):

    
    
    # iterate from parameters
    iterationSize, populationSize, crossoverPc, mutationPm = parameters

    pathResult = "../output/Communities/"

    if file_path.__contains__('.paj'):
        pathResult = pathResult + file_path.replace('.paj', '')
    else:
        pathResult = pathResult + file_path.replace('.dat', '')
    if not os.path.exists(pathResult):
        os.makedirs(pathResult)
        print("The new directory is created!: " + pathResult)

    arguments = []   
    for item in range(0, len(populationSize)):
        I = int(iterationSize[item])
        N = int(populationSize[item])
        Pc = crossoverPc[item]
        Pm = mutationPm[item]

        parametrization = "I_" + str(I) + "_N_" + str(N) + "_Pc_" + str(Pc) + "_Pm_" + str(Pm) + "_item_" + str(item + 1)
        print("Begin Parametrization: " + parametrization)

        arguments.append((Gi, N, I, 1.5, Pc, Pm, parametrization))

    with mp.Pool(mp.cpu_count()) as pool:  
        results = pool.starmap(wrapper, arguments)

    #graphClustering = GA.ga_community_detection(Gi, N, I, 1.5, Pc, Pm)


    
    for c, p in results:
        fileName = open(pathResult + "/" + p, 'w')
        for ci in c:
            ci.sort()
            ci = str(ci).replace(',', '').replace('[', '').replace(']', '')
            fileName.write(ci)
            fileName.write('\n')
        print("End Save File Parametrization: " + p)
        print("-----------------------------------------------")
        print('\n')
        fileName.close()
    print("End Network: " + file_path)
    print('\n')
    print('\n')

def create_graph():

    path = "../data/BD/LFRNetsTest/"
    for file in os.listdir(path):        
        # Check whether file is in text format or not
        if file.endswith(".dat") or file.endswith(".paj"):
            # get the Network File
            file_path = f"{file}"
            print("Begin: " + file_path)
            edgeList = []
            nodesList = []
            # read net file and get nodes and edges
            with open(path + file_path) as f:
                line = f.readline()
                while not line.__contains__("Edges"):
                    line = f.readline()
                for line in f:
                    if not line.__contains__("Vertices") and not line.__contains__("Edges"):
                        for x in line.split():
                            v_i = int(x) - 1
                            if not nodesList.__contains__(v_i):
                                nodesList.append(v_i)
                        edgeList.append(tuple([int(x) - 1 for x in line.split()]))

            Gi = nx.Graph()
            Gi.add_nodes_from(nodesList)
            Gi.add_edges_from(edgeList)

    return Gi

# G = nx.karate_club_graph()
# community_detection(G.nodes, G.edges, 10, 30, 1.5, 0.9, 0.1)

print_hi('GA_Net')

if __name__ == '__main__':
    star = time.time()
    print('begin time: ', star)
    params = readParameter()

    Gi = create_graph()

    run(Gi, parameters= params , file_path= 'network7.dat')

    end = time.time()
    print('end time: ', end)
    print('execution time: ',  end - star)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
