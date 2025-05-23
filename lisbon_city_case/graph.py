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
        popu_list = [ 5092,  8912,  5344,  5850,     0,  5838,  7263, 18623,     0,
       17202, 13461,     0,     0,     0,     0,     0,     0,  5426,
        2043,   923,     0,  9276, 17316, 10288,     0,  2652, 14373,
        3697,  4109,     0,  5776,     0,  2353,     0,     0,     0,
           0,  6386,  9065, 19164,     0, 16138,  4031,  2614,  3434,
           0,  2586,     0,  5829,  8285,   434,   100,     0,     0,
           0,     0,     0,     0,  8018,  3242,  6891,     0,  9037,
           0,  4234,  9066,   129,    21,     0,   132,  3539, 11163,
           0,  5411,     0,     0,     0,     0,     0,     0,  1432,
        8213,  1908,   554,     0,   466, 10698,  7671,     0,  8263,
        3968,  4322, 14243,     0, 12696,     0,     0,     0,     0,
       10606,     0,  4792,  6292, 11117,     0, 10553,  5354,  1763,
        8843,     0, 12799,     0,     0, 11306, 16417,     0,     0,
           0,     0,     0,     0,  5571, 10386,  9543,  7229,     0,
        9596,     0,     0, 10192, 12271,  5554,     0,  5173,  6942,
         238,     0,     0,     0,     0,  6650,     0,  8225,     0,
           0,  6540,  7485,  9103,     0, 10476, 10104, 11480,     0,
        1816,  4294, 10080,     0,     0,     0,     0,     0,  4847,
       16474, 21147,     0, 20752, 12947,  6498,     0,  3631,   132,
          81,  2614,     0,  2429,     0,     0,     0,     0,     0,
           0,     0,  8971,  6968,     0,  7770,  3614,  2908,  3663,
           0,  1944,     0]
        
        
        POI_list = [ 212,  109,   42,   25,    4,   28,  106,  104,   78,    8,    4,
         10,   75,   50,   47,   78,  108,  136,   63,   48,   54,   43,
          8,    1,   58,   74,   70,   50,   44,   83,   76,   15,   27,
         13,   20,    3,  278,  167,   21,   11,   24,   32,  230,   93,
          8,    5,    2,    1,  120,  153,   90,   41,    2,  107,  182,
         30,   58,   49,   54,   24,   80,   98,   53,   25,   11,  210,
        316,   78,   18,    2,   62,   29,   90,  132,    9,    9,   67,
        242,  147,  190,   46,    8,  195,   37,   16,  189,  412,  186,
        104,  159,  115,  165,   60,   22,  144,   23,    7,  258,  356,
        418,  287,  125,  136,  108,  377,   60,   12,   28,    0,  729,
        539,  679,  742,  310,  266,   57,   83,   66,   52,   22,    0,
       1125,  808,  412,  734,  568,  619,   68,    1,    3,   13,   15,
          0,  438,  434,  461,  665,  269,   84,   37,   88,    7,   26,
         11,    0,   12,  223,  134,   93,   35,   22,   77,  133,   52,
        167,   29,    0,    0,    6,   88,   19,   25,   60,  134,  203,
         95,   16,   19,    0,    0,    0,   10,   50,   81,   35,   54,
        155,  191,   67,    9,    0,    0,    0,    0,    0,  128,  126,
        228,  332,  265,   38,   19]
        
   
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


    
    
    
    
    