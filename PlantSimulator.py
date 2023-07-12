import numpy as np
from collections import deque
class Simulator():
    # Defines which expansion direction that each gene in the genome corresponds to
    # (shaped using whitespace for visual clarity, it is a 1x9 array)
    GENOME_DIRECTION_TRANSLATOR = np.array(
        [(-1,-1), (-1, 0), (-1, 1),
        ( 0,-1), ( 0, 0), ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)])

    # Define Materials to increase code readability
    AIR = np.int32(0)
    PLANT = np.int32(2)
    DIRT = np.int32(1)
    ROOT = np.int32(3)

    def __init__(self):
        self.steps = 50
        self.w_cols = 15
        self.w_rows = 15
        self.dirt_rows = np.int(0.3 * self.w_rows)

        self.sunlight_increment = 1
        self.water_increment = 1
        
        self.cell_sunlight_capacity = 1
        self.cell_water_capacity = 1

        self.cell_sunlight_consumption = 0.5
        self.cell_water_consumption = 0.5
        return
    
    ''' calculate_fitness
    Arguments:
        genome: (1 x N) numpy array where N = GENOME_LENGTH
    Returns:
        fitness_value: Floating point number representing the fitness of the genome
    '''
    def calculate_fitness(self, genome):
        per_gene_fitness = [-4, -4, -4, 12,  0, 12, -4, -4, -4,
                            0,  0,  0,  0,  0,  0,  0,  0,  0]
        
        fitness_value = (per_gene_fitness * genome).sum()

        return fitness_value
    
    '''Returns a normalized genome with genes corresponding to invalid directions 
    set to zero
    Valid growth directions are determined by two rules:
    1. If the direction is diagonal, each of the prospective cell's neighbors
        except for the parent cell need to == the target
    2. If the direction is cardinal, each of the prospective cell's neighbors 
        except for the parent cell and the parent cell's neighbors in other 
        cardinal directions need to == the target
    '''
    def norm_pos_genome(self, array, row, col, target, genome):
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

    def simulate(self, genome, rng):

        stack = deque()

        world = np.ndarray((self.w_rows, self.w_cols), dtype=np.int32)
        # Fill bottom with dirt and top with air
        world[0:self.dirt_rows, :].fill(self.DIRT)
        world[self.dirt_rows:self.w_rows, :].fill(self.AIR)
        # Plant a seed (ROOT with PLANT on top at the dirt line)
        world[self.dirt_rows-1, int(self.w_cols/2)] = self.ROOT
        stack.append((self.ROOT,(self.dirt_rows-1, int(self.w_cols/2))))
        world[self.dirt_rows, int(self.w_cols/2)] = self.PLANT
        stack.append((self.PLANT,(self.dirt_rows, int(self.w_cols/2))))

        stored_sunlight = 0.0
        stored_water = 0.0

        # Perform the simulation
        for step in range(self.steps):
            # Collect Resources
            # Collect Sunlight: for each column in the world, increase 
            # stored_sunlight by sunlight_increment if it contains PLANT
            for col in range(self.w_cols):
                if (world[:,col] == self.PLANT).sum() > 0:
                    stored_sunlight += self.sunlight_increment

            # Collect Water: for each DIRT in the world, give the plant water
            # if there is at least one ROOT in 3x3 neighborhood around it
            for row in range(self.dirt_rows):
                for col in range(self.w_cols):
                    if (world[row, col] == self.DIRT) \
                    and ((world[row-1:row+2, col-1:col+2] == self.ROOT).sum() > 0):
                        stored_water += self.water_increment
            
            # Apply stored water and sunlight limits before consumption
            stored_sunlight = min(stored_sunlight,len(stack) * self.cell_sunlight_capacity)
            stored_water = min(stored_water, len(stack) * self.cell_water_capacity)

            # Consume Resources
            for cell in range(len(stack)):
                #print('CONSUME RESOURCES TIME')
                if stored_sunlight >= self.cell_sunlight_consumption \
                and stored_water >= self.cell_water_consumption:
                    # consume resources
                    stored_sunlight -= self.cell_sunlight_consumption
                    stored_water -= self.cell_water_consumption
                else:
                    #print("CELL DEATH")
                    # The most recent cell is popped from the stack and removed
                    cell = stack.pop()
                    world[cell[1]] = self.AIR if (cell[0] == self.PLANT) else self.DIRT

            # Attempt to grow a child cell from each living cell
            for cell in range(len(stack)):
                c_row = stack[cell][1][0]
                c_col = stack[cell][1][1]
                material = stack[cell][0]
                if material == self.PLANT or material == self.ROOT:
                    # The material that a new call can replace
                    target = self.AIR if material == self.PLANT else self.DIRT

                    # Get normalized positional genome at world[c_row, c_col]
                    if material == self.PLANT:
                        positional_genome = self.norm_pos_genome(world, c_row, c_col, target, genome[0:9])
                    else:
                        positional_genome = self.norm_pos_genome(world, c_row, c_col, target, genome[9:])

                    # If there is no valid direction to grow, 
                    # continue to next cell in stack
                    if positional_genome.sum() == 0:
                        continue

                    # Use the effective genome to radomly select a directional 
                    #   offset from the pregenerated list
                    # Offset has the form (row_offset, col_offset)
                    offset = rng.choice(a=self.GENOME_DIRECTION_TRANSLATOR,
                                        p=positional_genome)

                    # Apply the selected offset to the coords of the current
                    # cell, then update world and stack
                    if offset[0] != 0 or offset[1] != 0:
                        #offset_coords = (c_row, c_col) + offset
                        offset_row = c_row + offset[0]
                        offset_col = c_col + offset[1]
                        stack.append((material, (offset_row, offset_col)))
                        world[offset_row, offset_col] = material
        # print(stack)
        # print(world)
        return self.calculate_fitness(genome)