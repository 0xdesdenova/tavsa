from deap import base
from deap import creator
from deap import tools

import random
import array
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from genetic_algorithm import vrp
from genetic_algorithm import elitism

# set the random seed:
random_seed = 42
random.seed(random_seed)

# create the desired vehicle routing problem using a traveling salesman problem instance:
problem_name = "tavsa"
depot_location = 0

# Genetic Algorithm constants:

crossover_probability = 0.9  # probability for crossover
mutation_probability = 0.2  # probability for mutating an individual
hall_of_fame_size = 30


def init_vrp(num_of_vehicles, atm_list):

    # create the desired vehicle routing problem using a traveling salesman problem instance:
    num_of_vehicles = num_of_vehicles
    atm_list = atm_list
    vrp_instance = vrp.VehicleRoutingProblem(problem_name, num_of_vehicles, depot_location, atm_list)

    toolbox = base.Toolbox()

    # define a single objective, minimizing fitness strategy:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # create the Individual class based on list of integers:
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

    # create an operator that generates randomly shuffled keys:
    toolbox.register("randomOrder", random.sample, range(len(vrp_instance)), len(vrp_instance))

    # create the individual creation operator to fill up an Individual instance with shuffled keys:
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

    # create the population creation operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # fitness calculation - compute the max distance that the vehicles covered
    # for the given list of atms represented by keys:
    def verp_distance(individual):
        return vrp_instance.get_max_distance(individual),  # return a tuple

    toolbox.register("evaluate", verp_distance)

    # Genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / len(vrp_instance))
    toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0 / len(vrp_instance))

    return toolbox, vrp_instance


# Genetic Algorithm flow:
def main(date, population_size, number_of_generations, num_of_vehicles, atm_list):
    population_size = population_size
    number_of_generations = number_of_generations

    toolbox, vrp_instance = init_vrp(num_of_vehicles, atm_list)

    start = time.time()
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=population_size)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(hall_of_fame_size)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.ea_simple_with_elitism(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability,
                                                         ngen=number_of_generations, stats=stats, halloffame=hof, verbose=True)

    # print best individual info:
    best = hof.items[0]
    routes = vrp_instance.get_routes(best)
    end = time.time()
    print("-- Execution Time = ", end - start)
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    print("-- Route Breakdown = ", routes)
    print("-- total distance = ", vrp_instance.get_total_distance(best))
    print("-- max distance = ", vrp_instance.get_max_distance(best))

    # plot statistics:
    min_fitness_values, mean_fitness_values = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(min_fitness_values, color='red')
    plt.plot(mean_fitness_values, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.savefig(f'{date}_optimization.png')

    # plot best route:
    for index, route in enumerate(routes):
        plt.figure(index + 2)
        vrp_instance.plot_data(route, atm_list)
        plt.savefig(f'{date}_route_{index}.png')

    indexed_routes = vrp_instance.get_routes(best)

    routes = []
    for route in indexed_routes:
        final_route = []
        for atm in route:
            final_route.append(atm_list[atm])

        routes.append(final_route)

    return {'route': routes, 'total_time': vrp_instance.get_total_distance(best)}

