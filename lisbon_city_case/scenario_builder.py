import numpy as np
import networkx as nx
import collections
import ptline
import matplotlib.pyplot as plt
import graph

def build_initial_graph(walking_speed):
    metro_stations_line_1 = [[ 8.52690178,  3.86853214],
       [ 9.44760178,  3.95983214],
       [10.70050178,  4.19413214],
       [11.90170179,  4.54943214],
       [12.47920179,  5.36453214],
       [12.88400179,  6.15103214],
       [13.12490179,  6.82093214],
       [13.85860179,  7.05193214],
       [14.45810179,  7.59313214],
       [14.09400179,  8.37413214],
       [12.73660179,  8.37743214],
       [11.31210179,  7.67673214]]
    
    metro_stations_line_2 = [[ 0.77300179,  5.87163214],
       [ 1.45500179,  6.56903214],
       [ 2.92790179,  6.76703214],
       [ 3.76170179,  6.97383214],
       [ 4.22480179,  6.61963214],
       [ 4.54270179,  6.03773214],
       [ 5.62070179,  5.62083214],
       [ 6.45120179,  5.46243214],
       [ 6.90880179,  4.66053214],
       [ 7.89880179,  4.27553214],
       [ 8.52690179,  3.86853214],
       [ 8.91190179,  3.33393214],
       [ 8.96690178,  2.98083214],
       [ 9.46410179,  2.25593214],
       [ 9.76550179,  1.89953214],
       [10.00860178,  1.27363214],
       [10.81380179,  0.92273214],
       [11.99520179,  1.64433214]]


    metro_stations_line_3 = [[ 6.35110179, 10.40693214],
       [ 6.51940179,  9.56213214],
       [ 7.87130179,  8.87683214],
       [ 7.90210179,  8.19043214],
       [ 8.32450179,  7.54803214],
       [ 8.06160178,  6.74613214],
       [ 7.92190179,  5.78693214],
       [ 9.12310179,  5.29633214],
       [ 9.28810179,  4.61543214],
       [ 9.44760179,  3.95983214],
       [ 9.25510179,  3.45493214],
       [ 8.96690179,  2.98083214],
       [ 8.39380179,  2.34063214]]
    
    
    metro_stations_line_4 = [[ 7.15850179,  6.74723214],
       [ 8.06160179,  6.74613214],
       [ 9.57740179,  5.98823214],
       [ 9.88540179,  5.41733214],
       [10.74450179,  4.77933214],
       [10.70050179,  4.19413214],
       [10.65540179,  3.74203214],
       [10.59160179,  3.10183214],
       [10.54980179,  2.68713214],
       [10.49040179,  2.06673214],
       [10.25280179,  1.67073214],
       [10.00860179,  1.27363214],
       [ 9.35520179,  0.80063214]]

    nb_of_staions_each_metro_line = [ len(metro_stations_line_1),len(metro_stations_line_2),len(metro_stations_line_3),len(metro_stations_line_4) ]
    cumsum_nb_of_staions_each_metro_line = np.cumsum( nb_of_staions_each_metro_line )
    
    connection_between_lines = [[0, 22], [1, 39], [2, 48], [24, 41], [27,54], [35,44]]

    g = nx.DiGraph(list_waiting_time=[], walking_speed=walking_speed)

    all_stations = metro_stations_line_1 +metro_stations_line_2 +metro_stations_line_3 +metro_stations_line_4
    nb_of_all_stations = len(all_stations)
    g.add_nodes_from([i for i in range(nb_of_all_stations)])
    all_edges = [(i,i+1) for i in range(nb_of_all_stations) if i!= 11 and i!= 29 and i!= 42 and i!=55] + connection_between_lines
    g.add_edges_from(all_edges)

    

    # metro dwell time for each station (hour)
    dwell_time_1 = list(np.array([2,2,2,2,2,2,2,2,2,2,3])/60) #3 min/60 = 1/20 h
    dwell_time_2 = list(np.array([2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,1])/60)
    dwell_time_3 = list(np.array([2,3,3,2,2,2,2,2,2,1,2,2])/60)
    dwell_time_4 = list(np.array([2,3,2,2,2,1,1,1,2,1,2,2])/60)

    node_ids_line_1 = [i for i in range(cumsum_nb_of_staions_each_metro_line[0])]
    node_ids_line_2 = [i for i in range(cumsum_nb_of_staions_each_metro_line[0],cumsum_nb_of_staions_each_metro_line[1])]
    node_ids_line_3 = [i for i in range(cumsum_nb_of_staions_each_metro_line[1],cumsum_nb_of_staions_each_metro_line[2])]
    node_ids_line_4 = [i for i in range(cumsum_nb_of_staions_each_metro_line[2],cumsum_nb_of_staions_each_metro_line[3])]
    list_waiting_time = [6/60,5/60,5/60,5/60] # 7.5 mins/60 = 0.125 hour
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
    connection_and_transfer_time = [(0, 22,list_waiting_time[1]+2/60), (22, 0,list_waiting_time[0]+2/60),
                                    (1, 39,list_waiting_time[2]+2/60),(39, 1,list_waiting_time[0]+2/60),
                                    (2, 48,list_waiting_time[3]+2/60),(48, 2,list_waiting_time[0]+2/60),
                                    (24, 41,list_waiting_time[2]+2/60),(41, 24,list_waiting_time[1]+2/60),
                                    (27, 54,list_waiting_time[3]+2/60),(54, 27,list_waiting_time[1]+2/60),
                                    (35, 44,list_waiting_time[3]+2/60),(44, 35,list_waiting_time[2]+2/60)]
    
    g.add_connection(connection_and_transfer_time)
    g.add_centroids()
    g.add_edge_between_centroids()
    g.add_edge_between_centroid_and_station()

    
    g.node_color=["green" for i in range(len(metro_stations_line_1))]+["orange" for i in range(len(metro_stations_line_2))]+["y" for i in range(len(metro_stations_line_3))]+["blue" for i in range(len(metro_stations_line_4))]
    g.node_size=[50 for i in range(56)]


    g.leftmost, g.rightmost, g.bottommost, g.upmost = g.find_limits()
    centr_id_matr, acc_matr_init = g.build_accessibility_matrix()
    g.all_stations = all_stations
    g.all_edges = all_edges
    g.centr_id_matr = centr_id_matr
    g.acc_matr = acc_matr_init
    g.pop_matr = g.build_population_matrix()
    g.poi_matr = g.build_poi_matrix()
    return g