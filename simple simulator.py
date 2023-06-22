import matplotlib
import matplotlib.pyplot as plt
import numpy as np

rows = 30
cols = 30
dirt_level = 10
steps = 100

AIR = np.int32(0)
PLANT = np.int32(2) # PLANT = AIR + 2 (to keep delta simple)

DIRT = np.int32(1)
ROOT = np.int32(3) # ROOT = DIRT + 2 (to keep delta simple)

#rng = np.random.default_rng(seed=int(input("Enter a seed (number): ")))
rng = np.random.default_rng(0)

world = np.ndarray(shape=[rows, cols], dtype=np.int32) # Stores what material is at each cell in the world

sunlight = np.ndarray(shape=[rows, cols], dtype=np.float32) # Stores the amount of stored "sunlight" in each cell
sunlight.fill(np.int32(0))
water = np.ndarray(shape=[rows, cols], dtype=np.float32) #Stores the amount of stored "water" in each cell
sunlight.fill(np.int32(0))

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
f, axarr = plt.subplots(2,2)
axarr[0,0].set_title('Tree')
im1 = axarr[0,0].imshow(
    np.flipud(world), 
    cmap=matplotlib.colors.ListedColormap(['white', 'saddlebrown', 'forestgreen', 'chocolate']), 
    interpolation='nearest',
    animated=True
)
plt.show(block=False)

# Enter the loop
for step in range(steps):
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
                
                if (world[exp_row, exp_col] == AIR):
                    #print('placing new plant node')
                    delta[exp_row, exp_col] = 2
                
            elif world[row, col] == ROOT:
                # It grows to an adjacent spot that is under the dirt line and not adjacent to any other ROOT (except parent)
                
                exp_row = row + rng.integers(low= -1, high=2)
                exp_col = col + rng.integers(low= -1, high=2)
                
                # Check for out of bounds (of the world, and of the dirt)
                if (not exp_col >= 0) or (not exp_col < cols):
                    continue #skip growing a root node from this node for this cycle
                
                # Check for ROOT in 3x3 neighborhood (minus parent cell, inclding current expansion target)
                #     as well as any future ROOT neighbors in the delta
                if ((world[exp_row-1:exp_row+2, exp_col-1:exp_col+2] == ROOT).sum() > 1) or ((delta[exp_row-1:exp_row+2, exp_col-1:exp_col+2] == 2).sum() > 0):
                    continue #not allowed to grow a node right next to an existing node
                    
                if (world[exp_row, exp_col] == DIRT):
                    #print('placing new plant node')
                    delta[exp_row, exp_col] = 2
    
    # Apply the Delta that was produced in this step
    world = world + delta
    delta.fill(0)

    # Update the plot to reflect the new state of the world after the Delta was applied
    im1.set_array(np.flipud(world))
    plt.pause(0.002)

input('Press enter when done')