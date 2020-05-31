import json
from math import sin, cos, sqrt, atan2

import matplotlib.pyplot as plt


class TravelingSalesmanProblem:
    """This class encapsulates the Traveling Salesman Problem.
    atm coordinates are read from an online file and distance matrix is calculated.
    The data is serialized to disk.
    The total distance can be calculated for a path represented by a list of atm indices.
    A plot can be created for a path represented by a list of atm indices."""

    def __init__(self, name, atm_list):
        """
        Creates an instance of a TSP

        :param name: name of the TSP problem
        """

        # initialize instance variables:
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0

        # initialize the data:
        self.__initData(atm_list)

    def __len__(self):
        """
        returns the length of the underlying TSP
        :return: the length of the underlying TSP (number of atms)
        """
        return self.tspSize

    def __initData(self, atm_list):
        """Reads the serialized data, and if not available - calls __create_data() to prepare it
        """

        # attempt to read serialized data:
        try:
            with open('genetic_algorithm/atm_data.txt') as json_file:
                atm_data_object = json.load(json_file)
            self.locations = [[atm_data_object[atm]['latitude'], atm_data_object[atm]['longitude']] for atm in atm_list]
            for atm in atm_list:
                if atm_data_object[atm]['latitude'] == 0.0 or atm_data_object[atm]['longitude'] == 0.0:
                    print(atm)
            for atm_origin in atm_list:
                distance_vector = []
                for atm_destination in atm_list:
                    if 'distance_vector' in atm_data_object[atm_origin]:
                        if atm_destination in atm_data_object[atm_origin]['distance_vector']:
                            distance_vector.append(int(atm_data_object[atm_origin]['distance_vector'][atm_destination]))
                        else:
                            delta_longitude = atm_data_object[atm_origin]['longitude'] - atm_data_object[atm_destination]['longitude']
                            delta_latitude = atm_data_object[atm_origin]['latitude'] - atm_data_object[atm_destination]['latitude']
                            a = (sin(delta_latitude / 2)) ** 2 + cos(atm_data_object[atm_destination]['latitude']) * cos(atm_data_object[atm_origin]['latitude']) * (sin(delta_longitude / 2)) ** 2
                            c = 2 * atan2(sqrt(a), sqrt(1 - a))
                            distance = 6373.0 * c

                            # Se convirtió distancia en tiempo, y se agregó ventana de servicio.
                            time = distance * 3 + atm_data_object[atm_origin]['service_time']
                            distance_vector.append(time)
                    else:
                        delta_longitude = atm_data_object[atm_origin]['longitude'] - atm_data_object[atm_destination]['longitude']
                        delta_latitude = atm_data_object[atm_origin]['latitude'] - atm_data_object[atm_destination]['latitude']
                        a = (sin(delta_latitude / 2)) ** 2 + cos(atm_data_object[atm_destination]['latitude']) * cos(atm_data_object[atm_origin]['latitude']) * (sin(delta_longitude / 2)) ** 2
                        c = 2 * atan2(sqrt(a), sqrt(1 - a))
                        distance = 6373.0 * c

                        # Se convirtió distancia en tiempo, y se agregó ventana de servicio.
                        time = distance * 3 + atm_data_object[atm_origin]['service_time']
                        distance_vector.append(time)
                self.distances.append(distance_vector)
        except (OSError, IOError):
            pass

        # set the problem 'size':
        self.tspSize = len(self.locations)


    def getTotalDistance(self, indices):
        """Calculates the total distance of the path described by the given indices of the atms

        :param indices: A list of ordered atm indices describing the given path.
        :return: total distance of the path described by the given indices
        """
        # distance between th elast and first atm:
        distance = self.distances[indices[-1]][indices[0]]

        # add the distance between each pair of consequtive atms:
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]

        return distance

    def plotData(self, indices):
        """plots the path described by the given indices of the atms

        :param indices: A list of ordered atm indices describing the given path.
        :return: the resulting plot
        """

        # plot the dots representing the atms:
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # create a list of the corresponding atm locations:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # plot a line between each pair of consequtive atms:
        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt