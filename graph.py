import numpy as np
import networkx as nx
import collections
import bus_line
import matplotlib.pyplot as plt

# seed = np.random.seed(120)

def edgeOfBetweenCenters(a):
    node_1,node_2,node_3,node_4 = a[0],a[1],a[2],a[3]
    edge_list =[(node_1,node_2,1.0/4.5),(node_2,node_1,1.0/4.5)
                ,(node_2,node_3,1.0/4.5),(node_3,node_2,1.0/4.5)
                ,(node_3,node_4,1.0/4.5),(node_4,node_3,1.0/4.5)
                ,(node_4,node_1,1.0/4.5),(node_1,node_4,1.0/4.5)
               ,(node_1,node_3,1.0*1.414/4.5),(node_3,node_1,1.0*1.414/4.5)
               ,(node_2,node_4,1.0*1.414/4.5),(node_4,node_2,1.0*1.414/4.5)]
    return edge_list

def edgeCenterAndBusStation(center,bus_pos,all_pos,waiting_time):
    bus_stop_name = list( bus_pos.keys())
    bus_pos_list = list( bus_pos.values() )
    center_pos = np.array( all_pos[center] )
    list_out = []
    for i in range(len(bus_stop_name)):
        stop_name = bus_stop_name[i]
        if np.linalg.norm(center_pos-np.array(bus_pos_list[i]))<=3:#小于3km,才能走
            dis_center_bus = np.linalg.norm(center_pos-np.array(bus_pos_list[i]))/4.5
            list_out.append( (center,stop_name,dis_center_bus+waiting_time[stop_name]) )
            list_out.append( (stop_name,center,dis_center_bus) )
    return list_out

class Graph:
    
    def __init__(self, list_connection,list_waiting_time):
        self.g = nx.DiGraph()
        self.bus_node = []
        self.bus_pos = {}
        self.center_node = []
        self.center_pos = []
        self.all_node = []
        self.all_pos = {}
        self.number_of_BusStations = 0
        self.bus_waiting_time = {}
        
        self.list_connection = list_connection
        self.list_waiting_time = list_waiting_time
        
        self.list_frequency = []
        
    def add_bus_line(self,bus_line):
        self.g.add_nodes_from(bus_line.bus_stop_list)
        self.g.add_weighted_edges_from(bus_line.line)#
        self.bus_node += bus_line.bus_stop_list
        self.bus_pos = {**self.bus_pos.copy(), **bus_line.bus_stop_pos}.copy()
        self.all_node = self.bus_node.copy()
        self.all_pos = self.bus_pos.copy()
        self.number_of_BusStations += len( bus_line.bus_stop_list)
        self.bus_waiting_time = {**self.bus_waiting_time.copy(), **dict.fromkeys(bus_line.bus_stop_list,bus_line.waiting_time)}.copy()
        
        
    def add_connection(self, list_connection ):
        self.g.add_weighted_edges_from(list_connection)
        
    def add_center(self):
        center_node = []
        center_pos = []
        
        self.center_to_pos = {}
        
        old_center_node = [ i+80 for i in range(0,500) ]
        old_center_pos = np.reshape([[ (i,j) for j in range(20) ] for i in range(25)],(500,2))
        
        for i in range(len(old_center_node)):
            if min(np.linalg.norm( np.array(old_center_pos[i])-np.array( list(self.bus_pos.values() )),axis=1 ))<3:
                center_node.append( old_center_node[i] )
                center_pos.append( old_center_pos[i] )
                self.center_to_pos[old_center_node[i]] = old_center_pos[i]

        
        self.g.add_nodes_from(center_node)
        self.center_node = center_node
        self.center_pos = center_pos
        
        
        self.all_node +=  center_node
        for i in range(len(self.center_node)):
            self.all_pos[  center_node[i] ] = tuple(center_pos[i]) 
            
    def add_edge_between_centers(self):
        ss = self.center_node.copy()
        old_point_list = [[i,i+20,i+21,i+1] for i in ss]
        
        point_list = []
        for point in old_point_list:
            if point[0] in self.center_node and point[1] in self.center_node and point[2] in self.center_node and point[3] in self.center_node:
                point_list.append(point)
        
        
        list_edge_a = []
        for point in point_list:
            list_edge_a+= edgeOfBetweenCenters(point)
        list_edge_a = list(set(list_edge_a))
        self.g.add_weighted_edges_from(list_edge_a)
    
    def add_edge_between_centerAnsBusStation(self):
        list_edge = []
        for i in self.center_node:
            list_edge +=  edgeCenterAndBusStation(i,self.bus_pos,self.all_pos,self.bus_waiting_time)
        self.g.add_weighted_edges_from(list_edge)
    
    def show(self):
        fig=plt.figure(figsize=(10,10))
        node_color=["r" for i in range(self.number_of_BusStations)] + ['b' for i in range(len(self.center_node)) ]
        node_size=[50 for i in range(self.number_of_BusStations)] + [10 for i in range(len(self.center_node)) ]
        nx.draw(self.g, self.all_pos, with_labels=True, node_color=node_color, node_size = node_size)
        plt.show()
        
    def get_acc(self):
        #centroid i from 40 to 139, population of i
        self.center_polulation = {}
        for i in range(len(self.center_node)):
            r = np.linalg.norm(np.array(self.center_pos[i])-np.array([12.5,10]))
            self.center_polulation[self.center_node[i]] = 1600/0.15*np.exp(-0.12*r)
        self.total_polulation = np.sum( list(self.center_polulation.values()) )
        #claculate acc
        self.center_to_acc = {}
        length =  nx.all_pairs_dijkstra_path_length(self.g)
        list_acc = []
        for i, dict_ in length:
            if i in self.center_node:
                acc_i = 0.0
                for j in dict_.keys():
                    if j in self.center_node and j != i:
                        acc_i = acc_i + self.center_polulation[j]/dict_[j]
                list_acc.append(acc_i)
                self.center_to_acc[i] = acc_i
        list_acc_0 = list_acc.copy()
        list_acc_0.sort()
        
        #Akinson
        y_mean = np.mean( list_acc )   
        sum_ = 0.
        for i in range( len(list_acc) ):
            sum_ = sum_ + list_acc[i]**(-1) 
        sum_ = 1 - (sum_/len(list_acc))**(-1)/y_mean
        
        #Pietra     
        sum_P = 0.
        for i in range( len(list_acc) ):
            sum_P = sum_P +  np.abs(list_acc[i]-y_mean)/y_mean
        sum_P = sum_P/(2* len(list_acc) )
        
        #Theil
        sum_T = 0.
        for i in range( len(list_acc) ):
            sum_T = sum_T +  list_acc[i]/y_mean*np.log( list_acc[i]/y_mean )
        sum_T = sum_T/len(list_acc)
        
        return [np.mean(list_acc_0),list_acc,sum_,sum_P,sum_T]
    