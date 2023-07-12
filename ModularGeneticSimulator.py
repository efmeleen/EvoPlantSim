import numpy as np
from PlantSimulator import Simulator
# Remember to do this when GUI is implemented for each different simulator
# plantsim = True
# if plantsim:
#     from PlantSimulator import Simulator
    
# Initialize Simulator
simulator = Simulator()

# Initialize random number generator
rng = np.random.default_rng()

# Evolution Parameters
generations = 10
genomes_per_gen = 20
crossover_probability = 0.7
mutation_probability = 0.05
mutation_low = -0.5
mutation_high = 0.5
survival_ratio = 0.4
GENOME_LENGTH = 18

''' normalize_genome_halves
Arguments:
    genomes: (N x M) numpy array, where N = genomes_per_gen and M = GENOME_LENGTH

Returns:
    normalized_genomes: (N x M) numpy array with each half of each row normalized

Normalizes (seperately) the first 9 and last 9 entries in each row of the
genomes array.
'''
def normalize_genome_halves(genomes):
    for g in range(genomes_per_gen):
        # FIRST HALF [0:9] (PLANT genome)
        # If the genome contains a negative number, shift all genes up by that magnitude (subtract negative)
        if (genomes[g, 0:9] < 0).sum() > 0:
            genomes[g, 0:9] -= genomes[g, 0:9].min()
        # Normalize the genome ONLY if the sum is not 0
        if genomes[g, 0:9].sum() != 0:
            genomes[g, 0:9] /= genomes[g, 0:9].sum()

        # SECOND HALF [9:] (ROOT genome)
        # If the genome contains a negative number, shift all genes up by that magnitude (subtract negative)
        if (genomes[g, 9:] < 0).sum() > 0:
            genomes[g, 9:] -= genomes[g, 9:].min()
        # Normalize the genome ONLY if the sum is not 0
        if genomes[g, 9:].sum() != 0:
            genomes[g, 9:] /= genomes[g, 9:].sum()

    return genomes

# Evolution Data
genomes = rng.random((genomes_per_gen, GENOME_LENGTH))
genomes = normalize_genome_halves(genomes)
fitness = np.zeros((genomes_per_gen))

''' create_next_generation
Arguments:
genomes: (N x M) numpy array, where N = genomes_per_gen and M = GENOME_LENGTH
fitness: (N x 1) numpy array, where N = genomes_per_gen

Returns: numpy array of same dimensions as genomes representing the next
         generation of genomes for the simulation

Performs culling, crossover, and mutation to generate the next generation given
the previous generation and the previous generation's fitness post-simulation.
'''
def create_next_generation(genomes, fitness):

    # Initialize child array
    child_genomes = np.zeros((genomes_per_gen, GENOME_LENGTH))

    # Cull: Remove the parents with fitness in the bottom 50%
    sorted_indices = fitness.argsort()
    # TODO: add a warning if genomes_per_gen*survival_ratio < 1
    num_surviving_parents = int(genomes_per_gen*survival_ratio)
    top_parents = (genomes[sorted_indices[::-1]])[0:num_surviving_parents, :]
    top_fitness = (fitness[sorted_indices[::-1]])[0:num_surviving_parents]

    # Shift all fitness up by the lowest negative value if one is present
    if top_fitness.min() < 0:
        top_fitness -= top_fitness.min()

    # Create a uniform distribution if every genome has zero fitness
    # if fitness.sum() == 0:
    #     fitness += 1
    
    #print('Top Fitness = ' + str(top_fitness))
    top_fitness /= top_fitness.sum() # Normalize so it can be used as pdf

    # Select Pairs
    #print('Unmodified Fitness = ' + str(fitness))
    #print('Probabilities = ' + str(top_fitness))
    parent_pairs = rng.choice(a=np.array(range(num_surviving_parents)), 
                        size=(genomes_per_gen,2), 
                        p=top_fitness)

    # Reproduce and perform crossover
    for c in range(genomes_per_gen):
        if rng.random() < crossover_probability:
            # perform crossover (single point)
            cross_point = rng.integers(1, GENOME_LENGTH)
            child_genomes[c, 0:cross_point] = top_parents[parent_pairs[c,0], 0:cross_point]
            child_genomes[c, cross_point:] = top_parents[parent_pairs[c,1], cross_point:]
        else:
            # Child is copy of parent 0
            child_genomes[c, :] = top_parents[parent_pairs[c, 0], :]

    # Perform Mutation on a Per-Gene Basis
    for c in range(genomes_per_gen):
        for g in range(GENOME_LENGTH):
            if rng.random() < mutation_probability:
                child_genomes[c,g] += rng.uniform(mutation_low, mutation_high)

    child_genomes = normalize_genome_halves(child_genomes)

    return child_genomes

for generation in range(generations):
    # Calculate the fitness of each genome
    for genome in range(genomes_per_gen):

        # Run simulation and return fitness instead of calculating fitness
        fitness[genome] = simulator.simulate(genomes[genome, :], rng)
        print('Finished simulation ' + str(genome+1) + ' in generation ' + str(generation+1))
        #fitness[genome] = calculate_fitness(genomes[genome, :])

    # TODO: store and plot average and maximum fitness for each generation
    print('Gen ' + str(generation) + ' Average Fitness = ' + str((fitness.sum())/genomes_per_gen))
    print('    Maximum Fitness = ' + str(fitness.max()))

    # Prepare for the next generation
    genomes = create_next_generation(genomes, fitness)
    fitness = np.zeros((genomes_per_gen)) # reset fitness array