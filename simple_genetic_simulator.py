# In this version of the algorithm, I use a genetic algorithm to select the 
# best plants over multiple generations
# The genome of a plant consists of two parts: ROOT genome and PLANT genome
#    Each ROOT or PLANT genome contains the following information:
#    8 values (that sum to 1) to determine the probability that a given cell 
#    grows a new cell in each of 8 neighboring cells

import sys
import numpy as np
from collections import deque

# Launch GUI if there script was run with no arguments
#if len(sys.argv) == 1:

# Initialize random number generator
#rng = np.random.default_rng(seed=int(input("Enter a seed (number): ")))
rng = np.random.default_rng()

# Evolution Parameters
generations = 100
genomes_per_generation = 100
GENOME_LENGTH = 18
time_steps_per_generation = 30
# Probability per generation that each genome experiences a mutation
mutation_probability = 0.1
# Lower (inclusive) and Upper (exclusive) bound for the magnitude of a mutation
mutation_range = (-0.1, 0.1)
# Probability that child genome is a combination of parents' instead of a copy
crossover_probability = 0.7
# Lower (inclusive) and Upper (exclusive) bounds for crossover point in genome
# (child genome = [parent1[0:cross_point], parent2[cross_point:GENOME_LENGTH]]
cross_point_bounds = (1,GENOME_LENGTH-1)

# Simulation Parameters
world_rows = 20
world_cols = 20
dirt_rows = int(0.3*world_rows) # How many rows of dirt in the world
sunlight_increment = 1
water_increment = 0.25
sunlight_consumption = 0.25
water_consumption = 0.25
stored_sunlight_per_cell = 1
stored_water_per_cell = 1

# Define Materials to increase code readability
AIR = np.int32(0)
PLANT = np.int32(2) # PLANT = AIR + 2 (to keep delta simple)
DIRT = np.int32(1)
ROOT = np.int32(3) # ROOT = DIRT + 2 (to keep delta simple)

stored_water = None
stored_sunlight = None
stacks = None
worlds = None
fitness = None
genomes = None
# Initialize the variables that are reset with each generation
def initialize_sim_vars() -> None:
    global stored_water
    stored_water = np.ndarray(genomes_per_generation)

    global stored_sunlight
    stored_sunlight = np.ndarray(genomes_per_generation)

    global stacks
    stacks = np.ndarray((genomes_per_generation), dtype=np.object)
    for i in range(len(stacks)):
        stacks[i] = deque()

    # Stores what element is a what position in each world
    global worlds
    worlds = np.ndarray(shape=[world_rows, world_cols, genomes_per_generation],
                        dtype=np.int32)
    worlds[0:dirt_rows, :, :].fill(DIRT)
    worlds[dirt_rows:world_rows, :, :].fill(AIR)

    # Fitness of each genome
    global fitness
    fitness = np.zeros((genomes_per_generation))

    # Plant a seed in the same position in each world
    # (A 'seed' is a ROOT with a PLANT in the cell directly above it)
    for w in range(genomes_per_generation):
        worlds[dirt_rows-1, int(world_cols/2), w] = ROOT
        stacks[w].append((ROOT,(dirt_rows-1, int(world_cols/2))))
        worlds[dirt_rows, int(world_cols/2), w] = PLANT
        stacks[w].append((PLANT,(dirt_rows, int(world_cols/2))))

    # Create the initial array of genomes. Each genome consists of two 8-value
    # sequences (PLANT and ROOT genome) that each sum to 1
    # TODO: Don't forget to re-normalize each half of each genome at the
    # start of each generation
    global genomes
    genomes = rng.random((genomes_per_generation, GENOME_LENGTH))
    return

initialize_sim_vars()

# Defines which expansion direction that each gene in the genome corresponds to
# (shaped using whitespace for visual clarity, it is a 1x9 array)
GENOME_DIRECTION_TRANSLATOR = np.array(
    [(-1,-1), (-1, 0), (-1, 1),
     ( 0,-1), ( 0, 0), ( 0, 1),
     ( 1,-1), ( 1, 0), ( 1, 1)])

# Returns a normalized genome with genes corresponding to invalid directions 
# set to zero
# Valid growth directions are determined by two rules:
#   1. If the direction is diagonal, each of the prospective cell's neighbors
#      except for the parent cell need to == the target
#   2. If the direction is cardinal, each of the prospective cell's neighbors 
#      except for the parent cell and the parent cell's neighbors in other 
#      cardinal directions need to == the target
def norm_pos_genome(array, row, col, target, genome):
    # Get the dimensions of the parent array
    rows, cols = np.shape(array)

    # Create the boundary mask by creating two coordinate arrays and applying
    # an offset so that (row, col) coords are at the center of the arrays
    # (The boundary mask will store whether the corresponding elements in 
    # row_coords and col_coords are within bounds in the array)
    row_coords, col_coords = np.indices((5,5))
    row_coords += row-2
    col_coords += col-2
    boundary_mask = np.logical_and(
        np.logical_and((0 <= row_coords), (row_coords < rows)), 
        np.logical_and((0 <= col_coords), (col_coords < cols)))

    # Initialize the target mask to all zeros
    # Then for each each element in the boundary_mask, if it is 1, set the 
    #     corresponding element in target_mask to 1 if the corresponding 
    #     element in array == the target
    target_mask = np.zeros((5,5), np.int32)
    for r in range(5):
        for c in range(5):
            if boundary_mask[r,c] == 1:
                target_mask[r,c] = np.int32(
                    array[row_coords[r,c], col_coords[r,c]] == target)
    # Target mask now contains for each corresponding element in array
    # (Is this element in bounds) AND (does this element == target)

    # Create the valid_neighbor_mask
    # valid_neighbor_mask stores whether or not each cell is a valid neighbor
    # for a child cell (out of bounds) OR (element==target (implies in bounds))
    valid_neighbor_mask = np.logical_or(np.logical_not(boundary_mask), 
                                        target_mask)

    #NOTE: valid_neighbor_mask now encodes whether each cell is a valid 
    #      neighbor for a newly grown cell, but does not encode whether any
    #      adjacent cell to the parent is a valid growth target

    # Initialize the direction_mask
    # The direction_mask uses boundary_mask and target_mask to determine if the
    # corresponding cell in array is a valid location
    #     For diagonal growth directions: 
    #         Does the 3x3 grid in valid_neighbor_mask centered on this 
    #         position sum to exactly 8 (parent cell not valid neighbor)
    #     For cardinal growth directions:
    #         Does the 2x3 (or 3x2) grid in valid_neighbor_mask "centered" on 
    #         this position sum to at least 6 (parent cell not valid neighbor,
    #         cells in perpendicular cardinal directions to child don't matter)
    #         (this allows for growth that is perpendicular to parent "branch")
    direction_mask = np.zeros((5,5))
    for r in range(1,4,1):
        for c in range(1,4,1):
            if (r == 2) and (c == 1): # Cardinal Case (col - 1)
                direction_mask[r,c] = np.int32(
                    valid_neighbor_mask[1:4, 0:2].sum() >= 6)
            elif (r == 2) and (c == 3): # Cardinal Case (col + 1)
                direction_mask[r,c] = np.int32(
                    valid_neighbor_mask[1:4, 3:].sum() >= 6)
            elif (c == 2) and (r == 1): # Cardinal Case (row - 1)
                direction_mask[r,c] = np.int32(
                    valid_neighbor_mask[0:2, 1:4].sum() >= 6)
            elif(c == 2) and (r == 3): # Cardinal Case (row + 1)
                direction_mask[r,c] = np.int32(
                    valid_neighbor_mask[3:, 1:4].sum() >= 6)                
            elif (valid_neighbor_mask[r-1:r+2,c-1:c+2].sum()) == 8: # Diagonal
                direction_mask[r,c] = 1

    #NOTE: direction_mask now encodes whether each cell around the center has 
    #      the right number of valid neighbors in the right positions but does 
    #      not encode whether each cell is a valid growth target, because 
    #      valid_neighbor_mask also doesn't encode that information

    # This encodes direction_mask with valid growth target info
    direction_mask = np.logical_and(direction_mask, target_mask)

    # Reshape the central 3x3 subsection of direction_mask into 1x9 genome mask
    # (central element is removed)
    # Apply the genome_mask to the genome, then re-normalize genome and return
    genome_mask = np.reshape(direction_mask[1:4, 1:4], 9)
    genome_mask[4] = 1 # NOTE: This is a bandaid to always allow no growth
    positional_genome = genome_mask*genome

    # To prevent returning NaN from dividing by zero, return before dividing if
    # all genes are 0
    if positional_genome.sum() == 0:
        return positional_genome

    # Normalize the genome and return it
    normalized_positional_genome = positional_genome / positional_genome.sum()
    return normalized_positional_genome

# Fitness for a simulation should never be zero when it ends
def update_fitness(sim, step) -> None:
    global fitness
    global worlds
    global time_steps_per_generation
    global world_cols
    global PLANT

    # Fitness is determined by how quickly the plant puts a PLANT in each column
    # Fitness is equal to the remaining steps in the simulation
    # Once the fitness has been set, it shouldn't be updated
    # (This is a quirk of this specific fitness function depending on step)
    # if(fitness[sim] == 0):
    #     cur_sum = 0
    #     for col in range(world_cols):
    #         if(worlds[:, col, sim] == PLANT).sum() > 0:
    #             cur_sum += 1

    #     if cur_sum == world_cols:
    #         fitness[sim] = time_steps_per_generation - step

    # Fitness is the sum of columns with plant for every time step
    # dividied by the number of plants in the world (encourage wide but not tall)
    # cur_sum = 0
    # for col in range(world_cols):
    #     if(worlds[:, col, sim] == PLANT).sum() > 0:
    #         cur_sum += 1
    # print('PLANT sum = ' + str((worlds[:, :, sim] == PLANT).sum()))
    # fitness[sim] += cur_sum/((worlds[:, :, sim] == PLANT).sum())

    plant_columns = 0
    for col in range(world_cols):
        if(worlds[:, col, sim] == PLANT).sum() > 0:
            plant_columns += 1

    plant_cells = (worlds[:,:,sim] == PLANT).sum()

    if plant_cells == 0:
        fitness[sim] += 0
    else:
        fitness[sim] += plant_columns/plant_cells

# Takes the current generation's genomes and corresponding fitness and returns
# the genomes that will be used in the next generation.
# NOTE: This function assumes that higher fitness is better
def produce_next_generation(current_genomes, current_fitness):
    global genomes_per_generation
    global crossover_probability
    global cross_point_bounds

    # Initialize child genome array
    child_genomes = np.ndarray((genomes_per_generation, GENOME_LENGTH))

    # Cloning and Crossover:"Roulette Wheel Selection","Single point crossover"
    # Fitness is normalized so it can be used as a probability distribution.
    # Parents are chosen based on that distribution. A parent can be chosen any
    # number of times.
    normalized_fitness = current_fitness/(current_fitness.sum())
    parent_pairs = rng.choice(a=np.array(range(genomes_per_generation)), 
                              size=(genomes_per_generation,2), 
                              p=normalized_fitness)
    # For each child genome, either clone a parent or perform crossover
    for i in range(genomes_per_generation):
        if rng.random() > crossover_probability:
            # Select a cross point within bounds, then set the each half of the
            # child genome to the corresponding halves of the parent genomes
            cross_point = rng.integers(low=cross_point_bounds[0], high=cross_point_bounds[1])
            child_genomes[i, 0:cross_point] = current_genomes[parent_pairs[i,0], 0:cross_point]
            child_genomes[i, cross_point:] = current_genomes[parent_pairs[i,1], cross_point:]
        else:
            # The child is a copy of the first parent
            child_genomes[i,:] = current_genomes[parent_pairs[i,0], :]
    
    # Mutate:
    # For each gene in each genome, decide if it will be mutated, then mutate
    for i in range(genomes_per_generation):
        for gene in range(GENOME_LENGTH):
            if rng.random() < mutation_probability:
                child_genomes[i,gene] += rng.uniform(mutation_range[0], mutation_range[1])

    # Normalize each genome (want all genomes to have same order of magnitude)
    for i in range(genomes_per_generation):
        # Normalize the first half
        child_genomes[i, 0:int(GENOME_LENGTH/2)] /= child_genomes[i, 0:int(GENOME_LENGTH/2)].sum()
        # Normalize the second half
        child_genomes[i, int(GENOME_LENGTH/2):] /= child_genomes[i, int(GENOME_LENGTH/2):].sum()

    return child_genomes

# Carry out the simulation for the specified number of generations
for generation in range(generations):
    # Carry out the simulation for each genome in its own seperate world
    # TODO: try switching the world and step loops to if it runs faster
    # TODO: consider putting the simulation for each world in its own thread to
    #       allow for parallel computation, then move on to the next generation
    #       once all simulations have concluded
    for sim in range(genomes_per_generation):
        # Get this simulation's corresponding world, genome, and stack
        world = worlds[:,:,sim]
        genome = genomes[sim,:]
        stack = stacks[sim]

        # print('Starting simulation ' + str(sim + 1) + ' in generation '
        #       + str(generation + 1))
        
        for step in range(time_steps_per_generation):
            # Collect Resources
            # Collect Sunlight: for each column in the world, increase 
            # stored_sunlight by sunlight_increment if it contains PLANT
            for col in range(world_cols):
                if (world[:,col] == PLANT).sum() > 0:
                    stored_sunlight[sim] += sunlight_increment

            # Collect Water: for each DIRT in the world, give the plant water
            # if there is at least one ROOT in 3x3 neighborhood around it
            for row in range(dirt_rows):
                for col in range(world_cols):
                    if (world[row, col] == DIRT) \
                    and ((world[row-1:row+2, col-1:col+2] == ROOT).sum() > 0):
                        stored_water[sim] += water_increment
            
            # Apply stored water and sunlight limits before consumption
            stored_sunlight[sim] = min(stored_sunlight[sim],len(stack)
                                       * stored_sunlight_per_cell)
            stored_water[sim] = min(stored_water[sim], 
                                    len(stack)*stored_water_per_cell)

            # Consume Resources
            for cell in range(len(stack)):
                #print('CONSUME RESOURCES TIME')
                if stored_sunlight[sim] >= sunlight_consumption \
                and stored_water[sim] >= water_consumption:
                    # consume resources
                    stored_sunlight[sim] -= sunlight_consumption
                    stored_water[sim] -= water_consumption
                else:
                    #print("CELL DEATH")
                    # The most recent cell is popped from the stack and removed
                    cell = stack.pop()
                    world[cell[1]] = AIR if (cell[0] == PLANT) else DIRT

            # Attempt to grow a child cell from each living cell
            for cell in range(len(stack)):
                c_row = stack[cell][1][0]
                c_col = stack[cell][1][1]
                material = stack[cell][0]
                if material == PLANT or material == ROOT:
                    # The material that a new call can replace
                    target = AIR if material == PLANT else DIRT

                    # Get normalized positional genome at world[c_row, c_col]
                    if material == PLANT:
                        positional_genome = norm_pos_genome(world, c_row, c_col, target, genome[0:9])
                    else:
                        positional_genome = norm_pos_genome(world, c_row, c_col, target, genome[9:])

                    # If there is no valid direction to grow, 
                    # continue to next cell in stack
                    if positional_genome.sum() == 0:
                        continue

                    # Use the effective genome to radomly select a directional 
                    #   offset from the pregenerated list
                    # Offset has the form (row_offset, col_offset)
                    offset = rng.choice(a=GENOME_DIRECTION_TRANSLATOR,
                                        p=positional_genome)

                    # Apply the selected offset to the coords of the current
                    # cell, then update world and stack
                    offset_coords = (c_row, c_col) + offset
                    offset_row = c_row + offset[0]
                    offset_col = c_col + offset[1]
                    stack.append((material, (offset_row, offset_col)))
                    world[offset_row, offset_col] = material

            # Update the fitness on each step
            update_fitness(sim, step)

        #print(worlds[:,:,sim])
        # print('Finished simulation ' + str(sim + 1) + ' in generation ' 
        #       + str(generation + 1))

    print('Generation ' + str(generation+1) + ' Average Fitness = ' + str(fitness.sum()/genomes_per_generation))
    # Simulation has concluded for the current generation.
    # Evaluate fitness
    # Perform genetic algorithm steps to get next generation based on fitness
    # Renormalize the genomes in the genomes array
    #print(world[:,:])
    genomes = produce_next_generation(genomes, fitness)

    #TODO: REMEMBER TO RESET EVERYTHING THAT NEEDS TO BE RESET FOR THE NEXT GEN
    initialize_sim_vars()
