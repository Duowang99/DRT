import numpy as np
import networkx as nx
import collections
import ptline
import matplotlib.pyplot as plt
import graph

def build_initial_graph(walking_speed):
    metro_stations_line_1 = [[3.74820692, 3.67372096],
       [4.13859691, 3.78273096],
       [4.30502692, 4.02572096],
       [4.69068692, 4.28620096],
       [5.13860692, 4.58518096],
       [5.50116692, 4.82883096],
       [5.83281692, 5.05136096],
       [6.17579692, 5.28027096],
       [6.64010692, 5.58475096],
       [7.14588692, 5.94742096],
       [8.19704692, 6.17908096]]
    
    metro_stations_line_2 = [[ 0.88941692,  4.08215096],
       [ 1.05364692,  4.86128096],
       [ 2.37430692,  4.78351096],
       [ 3.21756692,  4.63886096],
       [ 4.10724692,  3.76788095],
       [ 5.05324692,  3.46340096],
       [ 5.91311692,  3.70331096],
       [ 7.29988691,  3.99778096],
       [ 9.88488692,  4.05179096],
       [11.42884692,  4.16509096],
       [13.13890692,  4.35407096]]


    metro_stations_line_3 = [[8.05657692e+00, 1.06807210e+01],
       [6.93534692e+00, 1.05387110e+01],
       [6.21506692e+00, 9.35379096e+00],
       [5.77902692e+00, 8.23058096e+00],
       [5.49478692e+00, 7.57146096e+00],
       [5.12265692e+00, 6.71203096e+00],
       [4.81432692e+00, 5.96810096e+00],
       [4.33769692e+00, 5.22373096e+00],
       [4.17940692e+00, 4.43635096e+00],
       [4.18072692e+00, 3.76766096e+00],
       [4.42976692e+00, 3.13285096e+00],
       [4.99417691e+00, 2.87655096e+00],
       [6.12959692e+00, 2.36747096e+00],
       [6.80972692e+00, 2.14857096e+00],
       [8.20078692e+00, 1.65280096e+00],
       [9.25689692e+00, 1.27308096e+00],
       [1.04640369e+01, 7.90840962e-01],
       [1.12551569e+01, 5.05940962e-01],
       [1.20680569e+01, 2.65040962e-01],
       [1.45470169e+01, 1.13809615e-02]]
    
    
    metro_stations_line_4 = [[0.44754692, 0.10070096],
       [1.85543692, 0.18507096],
       [3.27223692, 1.12414096],
       [3.42392692, 1.53675096],
       [4.23693692, 2.16694096],
       [4.56396692, 2.51124096],
       [4.99615692, 2.84960096],
       [6.15676692, 3.23625096],
       [6.73074692, 3.60123096],
       [7.17041692, 4.08600096]]

    nb_of_staions_each_metro_line = [ len(metro_stations_line_1),len(metro_stations_line_2),len(metro_stations_line_3),len(metro_stations_line_4) ]
    cumsum_nb_of_staions_each_metro_line = np.cumsum( nb_of_staions_each_metro_line )
    
    connection_between_lines = [[1,15],[1,31],[15,31],[18,51],[33,48]]

    g = nx.DiGraph(list_waiting_time=[], walking_speed=walking_speed)

    all_stations = metro_stations_line_1 +metro_stations_line_2 +metro_stations_line_3 +metro_stations_line_4
    nb_of_all_stations = len(all_stations)
    g.add_nodes_from([i for i in range(nb_of_all_stations)])
    all_edges = [(i,i+1) for i in range(nb_of_all_stations) if i!= 10 and i!= 21 and i!= 41 and i!=51] + connection_between_lines
    g.add_edges_from(all_edges)

    

    # metro dwell time for each station (hour)
    dwell_time_1 = list(np.array([1,2,1,1,1,1,1,1,2,2])/60) #3 min/60 = 1/20 h
    dwell_time_2 = list(np.array([2,2,2,2,2,2,2,2,2,2])/60)
    dwell_time_3 = list(np.array([2,2,2,2,2,2,2,2,2,12,1,2,1,2,2,2,1,1,2])/60)
    dwell_time_4 = list(np.array([2,2,1,2,1,1,2,1,1])/60)

    node_ids_line_1 = [i for i in range(cumsum_nb_of_staions_each_metro_line[0])]
    node_ids_line_2 = [i for i in range(cumsum_nb_of_staions_each_metro_line[0],cumsum_nb_of_staions_each_metro_line[1])]
    node_ids_line_3 = [i for i in range(cumsum_nb_of_staions_each_metro_line[1],cumsum_nb_of_staions_each_metro_line[2])]
    node_ids_line_4 = [i for i in range(cumsum_nb_of_staions_each_metro_line[2],cumsum_nb_of_staions_each_metro_line[3])]
    list_waiting_time = [3/60,4/60,4/60,4/60] # 7.5 mins/60 = 0.125 hour
    #create bus_line
    metro_line_1 = ptline.PTline( '0',list_waiting_time[0],
                                     node_ids_line_1,
                                     dict( zip(node_ids_line_1, metro_stations_line_1)),
                                     dwell_time_1 )
    
    
    metro_line_2 = ptline.PTline( '1',list_waiting_time[1],
                                     node_ids_line_2,
                                     dict( zip(node_ids_line_2, metro_stations_line_2)),
                                     dwell_time_2 )
    
    metro_line_3 = ptline.PTline( '2',list_waiting_time[2],
                                     node_ids_line_3,
                                     dict( zip(node_ids_line_3, metro_stations_line_3)),
                                     dwell_time_3 )
    
    metro_line_4 = ptline.PTline( '3',list_waiting_time[3],
                                     node_ids_line_4,
                                     dict( zip(node_ids_line_4, metro_stations_line_4)),
                                     dwell_time_4 )


    
    #create Public transit graph
    g  = graph.Graph( list_waiting_time, walking_speed = walking_speed )
    
    #add each bus_line
    g.add_metro_line(metro_line_1)
    g.add_metro_line(metro_line_2)
    g.add_metro_line(metro_line_3)
    g.add_metro_line(metro_line_4)
    
    #add transfer station and time  (7,47,list_waiting_time[1]+2/60) means from line_1 (station 7) tansfer to line_2 (station 47),     -----[[0, 22], [1, 39], [2, 48], [24, 41], [27,54], [35,44]]
    #                                the time cost is average waiting time of line 2 + 2 mins of walking
    connection_and_transfer_time = [(1, 15,list_waiting_time[1]+2/60), (15, 1,list_waiting_time[0]+2/60),
                                    (1, 31,list_waiting_time[2]+2/60),(31, 1,list_waiting_time[0]+2/60),
                                    (15, 31,list_waiting_time[2]+2/60),(31, 15,list_waiting_time[1]+2/60),
                                    (18, 51,list_waiting_time[3]+2/60),(51, 18,list_waiting_time[1]+2/60),
                                    (33, 48,list_waiting_time[3]+2/60),(48, 33,list_waiting_time[2]+2/60)]

    
    g.add_connection(connection_and_transfer_time)
    g.add_centroids()
    g.add_edge_between_centroids()
    g.add_edge_between_centroid_and_station()

    
    g.node_color=["yellow" for i in range(len(metro_stations_line_1))]+["r" for i in range(len(metro_stations_line_2))]+["blue" for i in range(len(metro_stations_line_3))]+["green" for i in range(len(metro_stations_line_4))]
    g.node_size=[50 for i in range(52)]


    g.leftmost, g.rightmost, g.bottommost, g.upmost = g.find_limits()
    centr_id_matr, acc_matr_init = g.build_accessibility_matrix()
    g.all_stations = all_stations
    g.all_edges = all_edges
    g.centr_id_matr = centr_id_matr
    g.acc_matr = acc_matr_init
    g.pop_matr = g.build_population_matrix()
    g.poi_matr = g.build_poi_matrix()
    
    return g