import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
 
stack = deque() # Stack used to determine the order in which resources are allocated

rows = 30
cols = 30
dirt_level = 10
steps = 200

sunlight_increment = 1.0
water_increment = 1.0

cell_water_consumption = 0.25
cell_sunlight_consumtion = 0.25

cell_water_storage_capacity = 0.5
cell_sunlight_storage_capacity = 0.5

stored_water = 0.0
stored_water_history = np.ndarray(shape=steps)
stored_water_history.fill(0.0)

stored_sunlight = 0.0
stored_sunlight_history = np.ndarray(shape=steps)
stored_sunlight_history.fill(0.0)

number_of_cells_history = np.ndarray(shape=steps)
number_of_cells_history.fill(0.0)

AIR = np.int32(0)
PLANT = np.int32(2) # PLANT = AIR + 2 (to keep delta simple)

DIRT = np.int32(1)
ROOT = np.int32(3) # ROOT = DIRT + 2 (to keep delta simple)

#rng = np.random.default_rng(seed=int(input("Enter a seed (number): ")))
rng = np.random.default_rng(0)

world = np.ndarray(shape=[rows, cols], dtype=np.int32) # Stores what material is at each cell in the world

# sunlight = np.ndarray(shape=[rows, cols], dtype=np.float32) # Stores the amount of stored "sunlight" in each cell
# sunlight.fill(np.int32(0))
# water = np.ndarray(shape=[rows, cols], dtype=np.float32) #Stores the amount of stored "water" in each cell
# sunlight.fill(np.int32(0))

delta = np.ndarray(shape=[rows, cols], dtype=np.int32) # For each cell, stores if there will be a PLANT or ROOT there on the next step
delta.fill(np.int32(0))

# Initialize world, bottom rows to dirt up to dirt_level and remaining rows to air
for row in range(rows):
    for col in range(cols):
        if row <= dirt_level:
            world[row, col] = DIRT
        else:
            world[row, col] = AIR

# Plant seed at a point (1 root node in the dirt and 1 plant node above it, roughly in the middle of the world)
world[dirt_level, int((cols/2))] = ROOT
world[dirt_level+1, int((cols/2))] = PLANT

# Set up the plot
# f, axarr = plt.subplots(2,2)
# axarr[0,0].set_title('Tree')
# axarr[1,0].set_title('Stored Sunlight')
# axarr[1,1].set_title('Stored Water')
# im1 = axarr[0,0].imshow(
#     np.flipud(world), 
#     cmap=matplotlib.colors.ListedColormap(['white', 'saddlebrown', 'forestgreen', 'chocolate']), 
#     interpolation='nearest',
#     animated=True
# )
# im2 = axarr[1,0].plot(stored_sunlight_history)
# im2 = axarr[1,1].plot(stored_water_history)
# plt.show(block=False)

# Enter the loop
for step in range(steps):
    # Resource Collection
    # Sunlight Collection: Starting at the top of each column, iterate downwards until a plant or dirt is found. Add 1 sunlight if PLANT
    # TODO: change this to just check if the row contains any PLANT at all (using numpy array slicing like for neighbors)
    for col in range(cols):
        for row in range((rows-1), dirt_level, -1):
            if world[row, col] == PLANT:
                stored_sunlight = stored_sunlight + sunlight_increment
                break

    # Water Collection
    # For each dirt cell, water_increment to stored_water if there is an adjacent ROOT
    for col in range (cols):
        for row in range(dirt_level, -1, -1):
            if world[row, col] == DIRT and ((world[row-1:row+2, col-1:col+2] == ROOT).sum() > 0):
                stored_water = stored_water + water_increment

    # Limit Stored Sunlight and Water to (Number of Cells) * (Cell capacity)
    stored_water = min(stored_water, (len(stack)+2)*cell_water_storage_capacity)
    stored_sunlight = min(stored_sunlight, (len(stack)+2)*cell_sunlight_storage_capacity)

    # Track the stored water and sunlight for later plotting
    stored_water_history[step] = stored_water
    stored_sunlight_history[step] = stored_sunlight

    # Resource Consumption
    # For each cell starting at the bottom of the stack, each cell consumes resources
    #   Once there aren't enough resources to feed the remaining cells, remove them from the world and pop them from the stack
    #   The stack implicitly stores the "seniority" of each cell, the popped cells are always the youngest
    for i in range(len(stack)):
        if (stored_sunlight >= cell_sunlight_consumtion) and (stored_water >= cell_water_consumption):
            stored_sunlight -= cell_sunlight_consumtion
            stored_water -= cell_water_consumption
        else:
            #print('PRUNING PRUNING PRUNING')
            for j in range(len(stack)-1, i, -1):
                if (stack[j][1][0] <= dirt_level):
                    world[stack[j][1]] = DIRT
                else:
                    world[stack[j][1]] = AIR
                stack.pop()
            break

    number_of_cells_history[step] = len(stack)

    # Growth
    for row in range(2,rows-2):
        for col in range(2, cols-2):
            if world[row, col] == PLANT:
                # it grows to an adjacent spot that is above the dirt and not adjacent to any other PLANT (except parent)
                
                exp_row = row + rng.integers(low= -1, high=2)
                exp_col = col + rng.integers(low= -1, high=2)
                
                # Check for out of bounds
                if (not exp_col >= 0) or (not exp_col < cols):
                    continue #skip growing a plant node from this node for this cycle
                    
                # Check for any non-air cell in neighborhood (minus parent cell, including current expansion target)
                #     or any future non-air cells in the delta
                if ((world[exp_row-1:exp_row+2, exp_col-1:exp_col+2] > 0).sum() > 1) or ((delta[exp_row-1:exp_row+2, exp_col-1:exp_col+2] == 2).sum() > 0):
                    continue #not allowed to grow a node right next to an existing node
                
                if (world[exp_row, exp_col] == AIR): #TODO: CHECK IF PLANT has enough resources to grow
                    #print('placing new plant node')
                    delta[exp_row, exp_col] = 2
                    stack.append((PLANT, (exp_row, exp_col)))
                
            elif world[row, col] == ROOT:
                # It grows to an adjacent spot that is under the dirt line and not adjacent to any other ROOT (except parent)
                
                exp_row = row + rng.integers(low= -1, high=2)
                exp_col = col + rng.integers(low= -1, high=2)
                
                # Check for out of bounds (of the world, and of the dirt)
                if (not exp_col >= 0) or (not exp_col < cols) or (not exp_row <= dirt_level):
                    continue #skip growing a root node from this node for this cycle
                
                # Check for ROOT in 3x3 neighborhood (minus parent cell, inclding current expansion target)
                #     as well as any future ROOT neighbors in the delta
                if ((world[exp_row-1:exp_row+2, exp_col-1:exp_col+2] == ROOT).sum() > 1) or ((delta[exp_row-1:exp_row+2, exp_col-1:exp_col+2] == 2).sum() > 0):
                    continue #not allowed to grow a node right next to an existing node
                    
                if (world[exp_row, exp_col] == DIRT): #TODO: CHECK IF PLANT has enough resources to grow
                    #print('placing new plant node')
                    delta[exp_row, exp_col] = 2
                    stack.append((ROOT, (exp_row, exp_col)))
    
    # Apply the Delta that was produced in this step
    world = world + delta
    delta.fill(0)

    # Update the plot to reflect the new state of the world after the Delta was applied
    # im1.set_array(np.flipud(world))
    # plt.pause(0.002)

f, axarr = plt.subplots(2,2)

axarr[0,0].set_title('Final Plant')
axarr[0,0].set_xticks([])
axarr[0,0].set_yticks([])
im1 = axarr[0,0].imshow(
    np.flipud(world), 
    cmap=matplotlib.colors.ListedColormap(['white', 'saddlebrown', 'forestgreen', 'chocolate']), 
    interpolation='nearest',
    animated=True
)

axarr[1,0].set_title('Stored Sunlight')
axarr[1,0].set_xlabel('Time Step')
axarr[1,0].set_ylabel('Stored Sunlight')
im2 = axarr[1,0].plot(stored_sunlight_history)

axarr[1,1].set_title('Stored Water')
axarr[1,1].set_xlabel('Time Step')
axarr[1,1].set_ylabel('Stored Water')
im3 = axarr[1,1].plot(stored_water_history)

axarr[0,1].set_title('Number of Cells')
axarr[0,1].set_xlabel('Time Step')
axarr[0,1].set_ylabel('Number of Cells')
im4 = axarr[0,1].plot(number_of_cells_history)

plt.tight_layout()
plt.show(block=True)