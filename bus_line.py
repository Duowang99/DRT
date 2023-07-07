import networkx as nx
import pylab 
import numpy as np
import matplotlib.pyplot as plt





class Bus_line:
    '''Parameters:  number : the line number of this bus line. e.g. 9106
                            waiting_time : average waiting_time
                            bus_stop_list : all the bus stop ids by a list
                            bus_stop_pos : all the bus stop positions by dict{ bus_stop_id: (x,y) }
    '''
    def __init__(self, number,waiting_time, bus_stop_list, bus_stop_pos,times):
        self.number = number
        self.bus_stop_list = bus_stop_list
        self.bus_stop_pos = bus_stop_pos
        self.waiting_time = waiting_time
        self.line_length = len( bus_stop_list ) -1
        
        self.times = times
        self.line = []
            
            #time可以直接读取
      
    
        #加上从start_id到stop_id所用时间
        for i in range(self.line_length):
            start_id = self.bus_stop_list[i]
            stop_id = self.bus_stop_list[i+1]
            time = self.times[i]
            self.line.append( ( start_id, stop_id, time ) )
            self.line.append( ( stop_id, start_id, time ) )
            #times.append(time)