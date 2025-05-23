import numpy as np
import networkx as nx
import collections
import ptline
import matplotlib.pyplot as plt
import copy

# seed = np.random.seed(120)

def edgeOfBetweenCentroids(a, walking_speed):
    node_1,node_2,node_3,node_4 = a[0],a[1],a[2],a[3]
    # distance between node_1 and node_2 is 1 km, and walking is 4.5 km/h, so time cost is 1/4.5 h.
    edge_list =[(node_1,node_2,1.0/walking_speed),(node_2,node_1,1.0/walking_speed)
                ,(node_2,node_3,1.0/walking_speed),(node_3,node_2,1.0/walking_speed)
                ,(node_3,node_4,1.0/walking_speed),(node_4,node_3,1.0/walking_speed)
                ,(node_4,node_1,1.0/walking_speed),(node_1,node_4,1.0/walking_speed)
               ,(node_1,node_3,1.0*1.414/walking_speed),(node_3,node_1,1.0*1.414/walking_speed)
               ,(node_2,node_4,1.0*1.414/walking_speed),(node_4,node_2,1.0*1.414/walking_speed)]
    return edge_list

def edgeCentroidAndStation(centroid,metro_pos,all_pos,metro_waiting_time, walking_speed):
    metro_station_name = list( metro_pos.keys())
    metro_station_list = list( metro_pos.values() )
    centroid_pos = np.array( all_pos[centroid] )
    list_out = []
    for i in range(len(metro_station_name)):
        station_name = metro_station_name[i]
        if np.linalg.norm(centroid_pos-np.array(metro_station_list[i]))<=3:#if distance between centroid and station < 3km, people can walk to station.
            time_cost_walk_centroid_station = np.linalg.norm(centroid_pos-np.array(metro_station_list[i]))/walking_speed
            list_out.append( (centroid,station_name,time_cost_walk_centroid_station+metro_waiting_time[station_name]) )
            list_out.append( (station_name,centroid,time_cost_walk_centroid_station) )
    return list_out


def compute_Akinson(list_):
    #Akinson
    y_mean = np.mean( list_ )   
    sum_ = 0.
    for i in range( len(list_) ):
        sum_ = sum_ + list_[i]**(-1) 
    sum_ = 1 - (sum_/len(list_))**(-1)/y_mean
    
    return 1- np.exp( np.mean( np.log(list_) )  )/y_mean

def compute_Pietra(list_):    
    #Pietra     
    sum_ = 0.
    y_mean = np.mean( list_ ) 
    for i in range( len(list_) ):
        sum_ = sum_ +  np.abs(list_[i]-y_mean)/y_mean
        sum_ = sum_/(2* len(list_) )
    return sum_
    
def compute_Theil(list_):    
    #Theil
    sum_ = 0.
    y_mean = np.mean( list_ ) 
    for i in range( len(list_) ):
        sum_ = sum_ +  list_[i]/y_mean*np.log( list_[i]/y_mean )
    sum_ = sum_/len(list_)
    return sum_

def compute_Gini(list_):  
    n_ = len(list_)
    list_.sort() #将会改变函数外list_的顺序，因为list_的是指针。
    sum_ = 0.
    for i in range( n_ ):
        sum_ = sum_ + (i+1)*list_[i]

    gini_ = 2*sum_/n_/np.sum(list_) - (n_+1)/n_
    return gini_


def Palma(list_):
    cp = int(len(list_)/10)
    return np.sum(list_[-1*cp:]) / np.sum(list_[:4*cp])



class Graph:
    
    def __init__(self, list_waiting_time, walking_speed):
        self.g = nx.DiGraph()
        
        self.metro_node = []
        self.metro_pos = {}
        
        self.centroid_node = []
        self.centroid_pos = []
        
        self.all_node = []
        self.all_pos = {}
        
        self.number_of_metro_stations = 0
        self.metro_waiting_time = {}
        self.list_waiting_time = list_waiting_time
        
        self.walking_speed = walking_speed

        self.node_color = []
        self.node_size = []

        self.all_stations = []
        self.all_edges = []
        self.centr_id_matr = np.zeros(shape=(1, 1)) # shape is temporary
        self.acc_matr = np.zeros(shape=(1, 1)) # shape is temporary
        self.pop_matr = np.zeros(shape=(1, 1)) # shape is temporary # Population matrix

        # These are the extreme x and y positions of all centroids
        self.leftmost=float('inf')
        self.rightmost=-float('inf')
        self.bottommost=float('inf')
        self.upmost=-float('inf')

        # Global indices
        self.ineq_Atkinson = 0
        self.ineq_Gini = 0
        self.avg_acc = 0
        
        
    def add_metro_line(self,metro_line):
        self.g.add_nodes_from(metro_line.metro_station_list)
        self.g.add_weighted_edges_from(metro_line.line)
        self.metro_node += metro_line.metro_station_list
        self.metro_pos = copy.deepcopy({**copy.deepcopy(self.metro_pos), **metro_line.metro_station_pos})
        self.all_node = copy.deepcopy(self.metro_node)
        self.all_pos = copy.deepcopy(self.metro_pos)
        self.number_of_metro_stations += len( metro_line.metro_station_list)
        self.metro_waiting_time = copy.deepcopy({**copy.deepcopy(self.metro_waiting_time), **dict.fromkeys(metro_line.metro_station_list,metro_line.waiting_time)})
        
        
    def add_connection(self, connection_and_transfer_time ):
        self.g.add_weighted_edges_from(connection_and_transfer_time)
        
    def add_centroids(self):
        centroid_node = []
        centroid_pos = []
        
        self.centroid_to_pos = {}
        
        old_centroid_node = [ i+80 for i in range(0,12*16) ]
        old_centroid_pos = np.reshape([[ (i,j) for j in range(12) ] for i in range(16)],(12*16,2))
        
        # Only centroids within 3 km from the bus station are kept.
        for i in range(len(old_centroid_pos)):
            if min(np.linalg.norm( np.array(old_centroid_pos[i])-np.array( list(self.metro_pos.values() )),axis=1 ))<3:
                centroid_node.append( old_centroid_node[i] )
                centroid_pos.append( old_centroid_pos[i] )
                self.centroid_to_pos[old_centroid_node[i]] = old_centroid_pos[i]

        
        self.g.add_nodes_from(centroid_node)
        self.centroid_node = centroid_node
        self.centroid_pos = centroid_pos
        
        
        self.all_node +=  centroid_node
        for i in range(len(self.centroid_node)):
            self.all_pos[  centroid_node[i] ] = tuple(centroid_pos[i]) 
            
    def add_edge_between_centroids(self):

        old_point_list = [[i,i+12,i+13,i+1] for i in copy.deepcopy(self.centroid_node)]
        
        point_list = []
        for point in old_point_list:
            if point[0] in self.centroid_node and point[1] in self.centroid_node and point[2] in self.centroid_node and point[3] in self.centroid_node:
                point_list.append(point)
        
        
        list_edge_a = []
        for point in point_list:
            list_edge_a+= edgeOfBetweenCentroids(point, self.walking_speed)
        list_edge_a = list(set(list_edge_a))
        self.g.add_weighted_edges_from(list_edge_a)
    
    def add_edge_between_centroid_and_station(self):
        list_edge = []
        for i in self.centroid_node:
            list_edge +=  edgeCentroidAndStation(i,self.metro_pos,self.all_pos,self.metro_waiting_time, self.walking_speed)
        self.g.add_weighted_edges_from(list_edge)
     
    
    def show(self):
        fig=plt.figure(figsize=(16,16))
        
        node_color=["r" for i in range(self.number_of_metro_stations)] + ['b' for i in range(len(self.centroid_node)) ]
        node_size=[50 for i in range(self.number_of_metro_stations)] + [10 for i in range(len(self.centroid_node)) ]
        nx.draw(self.g, self.all_pos, with_labels=True, node_color=node_color, node_size = node_size)
        plt.show()
        
    def compute_accessibility(self):
        
        #len(popu_list) = 500
        popu_list = [ 6282,     0,     0,  8672,     0,  6415,  6837,  6836,     0,
           0,  2598,     0,     0,  5885,  6472,  9671,     0,     0,
           0,     0,  5850,  3285,  4348,     0,  8208,  9436,     0,
           0,     0, 10854,  9894,  4958,  9173,     0,     0,     0,
           0,     0,  7553,  4510,     0,  6506,  7376,     0,     0,
       11061,  9363,     0,  6873, 14377,  4897, 12981,     0,     0,
           0,  2612,  1210,  2246,   986,     0,  2877,  3829,     0,
           0,     0, 19548, 17075, 19749,  9182,     0,     0,     0,
           0,     0, 12919, 23722,     0, 26426,     0,     0,     0,
        5443,  1856,     0,  1387,  7702, 22848,     0,     0,     0,
       14649, 12320, 16221,  8439,     0,     0,  3253,     0,     0,
       22345,     0, 25786,  8567,  6556,     0,     0,  9610,     0,
           0,  6208, 10455,  4710,     0, 10728,     0,     0, 10699,
        9900,  7638,     0,  3382,  3001,     0,     0,     0,     0,
        8886,  9367,  6436,     0,     0,     0,     0,     0,  6747,
        4268,     0,  9174,  7879,     0,     0,  2525,  7953,     0,
        7341,  3091,  2415,  4013,     0,     0,     0,  9632,  8958,
        6367,  8420,     0,  4205,  3038,     0,     0,     0,  5043,
        7429,  7834,  7772,     0,     0,     0,     0,     0,  4207,
        1841,     0,  5716,     0,     0,     0,  8612,  8054,     0,
        3668,  4849,  4959,     0,     0,     0,  9052,  7887,  3928,
        6677,     0,     0]
        
        
        POI_list = [ 523,   87,   53,  250,  123,  228,   56,   88,   13,    3,    8,
          3,  703,  193,  118,  345,  490,  956,  164,   77,   41,  215,
         44,   64,  939,  380,  278,  194,  601,  710,  307,  419,  297,
        225,   64,   16,  405,  895,  443,  287,  261,  364,  195,  140,
        429,  372,  182,   32,  204,  785,  771,  890, 1132,  882,  775,
        165,   96,    7,   52,  182,  280,  483,  310, 1155, 1102,  717,
        507,  317,  339,   70,   11,   19,   17,  188,  629,  752,  770,
        379,  154,  199,  314,  182,  116,   24,   47,  276,  389,  295,
        609,  501,  268,  303,  236,  140,  147,   91,    9,   27,  179,
        105,  235,  530,  163,   59,   85,  113,   93,  242,   14,   64,
         92,   10,   48,  208,  116,  139,   11,   43,  106,  226,   44,
         52,   82,   42,  130,   94,  234,   95,   59,   22,   19,   68,
        204,   68,   70,   55,   55,   74,  243,   39,   62,   74,   81,
        124,   89,   13,  103,   26,   40,  136,   96,   74,   71,   44,
         39,   67,   22,   12,   81,   67,  152,  117,  106,   43,  151,
        114,   30,   14,  135,   22,   63,   12,   83,  266,   26,   14,
         49,  228,   52,   24,   79,   98,   50,   10,   23,   62,   26,
         17,    3,   35,    1,    6]

   
        self.centroid_population = {}
        for i in range(len(self.centroid_node)):
            self.centroid_population[self.centroid_node[i]] = popu_list[self.centroid_node[i]-80]
        self.total_population = np.sum( list(self.centroid_population.values()) )
        
        self.centroid_POI = {}
        for i in range(len(self.centroid_node)):
            self.centroid_POI[self.centroid_node[i]] = POI_list[self.centroid_node[i]-80]
        self.total_POI = np.sum( list(self.centroid_POI.values()) )
        
        
        #claculate acc
        self.centroid_to_acc = {}
        length =  nx.all_pairs_dijkstra_path_length(self.g)
        list_acc = []
        list_acc_popul = []
        for i, dict_ in length:
            if i in self.centroid_node:
                acc_i = 0.0
                for j in dict_.keys():
                    if j in self.centroid_node and j != i:
                        acc_i = acc_i + self.centroid_POI[j]/dict_[j]
                list_acc.append(acc_i)
                self.centroid_to_acc[i] = acc_i
                
                for k in range(self.centroid_population[i]):
                    list_acc_popul.append(acc_i)
                
        list_acc_0 = copy.deepcopy(list_acc_popul)
        list_acc_0.sort()
        
        self.list_acc_0 = copy.deepcopy(list_acc_0)
        
        #Akinson 
        self.ineq_Atkinson = compute_Akinson(list_acc_0) #一定是注意代入的是list_acc_0，因为compute_Akinson会改变原有list的顺序带入list_acc就错了
        
        ##Pietra     
        self.sum_P = compute_Pietra(list_acc_0)
        
        #Theil
        self.ineq_Theil = compute_Theil(list_acc_0)

        self.avg_acc = 0
        cumulative_population = 0
        for c in self.centroid_node:
        #    self.avg_acc = self.avg_acc + self.centroid_population[c]*self.centroid_to_acc[c]
            cumulative_population = cumulative_population + self.centroid_population[c]
        self.avg_acc = np.mean(list_acc)
          
            
        self.palma = Palma(list_acc_0)
            
        
        return [np.mean(list_acc_0),list_acc, self.ineq_Atkinson,self.ineq_Theil,self.sum_P,self.palma]
    
    
    def find_limits(self):
        """
        Returns centroids the limits of the x- and y-positions of the
        centroids
        """
        leftmost=float('inf')
        rightmost=-float('inf')
        bottommost=float('inf')
        upmost=-float('inf')
        for pos in self.centroid_pos:
            leftmost = min(leftmost, pos[0])
            rightmost = max(rightmost, pos[0])
            bottommost = min(bottommost, pos[1])
            upmost = max(upmost, pos[1])
        
        return leftmost, rightmost, bottommost, upmost

    def build_accessibility_matrix(self):
        
        self.compute_accessibility()
        
        rows = self.upmost-self.bottommost+1
        cols = self.rightmost-self.leftmost+1
        acc_matr = np.array([([float('nan')]*cols) for i in range(rows)])
        centr_id_matr = np.array([([float('nan')]*cols) for i in range(rows)])
        
        for centr in self.centroid_node:
            acc = self.centroid_to_acc[centr]
            pos = self.centroid_to_pos[centr]
            centr_id_matr[ self.upmost-pos[1], pos[0] ]=centr
            acc_matr[ self.upmost-pos[1], pos[0] ]=acc
        
        return centr_id_matr, acc_matr

    def build_population_matrix(self):
        
        rows = self.upmost-self.bottommost+1
        cols = self.rightmost-self.leftmost+1
        pop_matr = np.array([([float('nan')]*cols) for i in range(rows)])
          
        for centr in self.centroid_node:
            population = self.centroid_population[centr]
            pos = self.centroid_to_pos[centr]
            pop_matr[ self.upmost-pos[1], pos[0] ]=population
        
        return pop_matr
    
    def build_poi_matrix(self):
        
        rows = self.upmost-self.bottommost+1
        cols = self.rightmost-self.leftmost+1
        pop_matr = np.array([([float('nan')]*cols) for i in range(rows)])
          
        for centr in self.centroid_node:
            population = self.centroid_POI[centr]
            pos = self.centroid_to_pos[centr]
            pop_matr[ self.upmost-pos[1], pos[0] ]=population
        
        return pop_matr

    #}end of Graph class


    
    
    
    
    