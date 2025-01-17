import numpy as np
import matplotlib.pyplot as plt


# Booth function implementation
def booth_function(X):
    x = X[:, 0]
    y = X[:, 1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def calculate_probabilities(f_values):
    reciprocal = 1 / f_values
    total = np.sum(reciprocal)
    probabilities = reciprocal / total
    return probabilities

# Update here -> a non uniform version of np.random.rand()
def roulette_wheel_selection(probabilities):
    if np.isclose(np.sum(probabilities), 1) == 0:
        probabilities = np.ones_like(f_values) / len(f_values)
    assert np.isclose(np.sum(probabilities), 1), "Probabilities sum != 1."
    cumulative_sum = np.cumsum(probabilities)
    random_value = np.random.rand()
    if random_value == 1:return len(probabilities) - 1
    else:
        selected_index = np.where(cumulative_sum > random_value)[0][0]
        return selected_index

# update her ->  best position instead of mean
def update_bounds(bounds, reduction_factor, best_position):
    new_bounds = []
    for i, (lower, upper) in enumerate(bounds):
        range_val = upper - lower
        reduced_range = range_val * reduction_factor
        
        # best position instead of mean
        center = best_position[i]
        half_reduced_range = reduced_range / 2
        new_lower = center - half_reduced_range
        new_upper = center + half_reduced_range
        
        new_bounds.append([
            max(lower, new_lower),
            min(upper, new_upper)
        ])
    return np.array(new_bounds)


# Parameters

# Parameters
# dimensions = 2
# candidates = 5  
# lower_bound, upper_bound = -10, 10
# reduction_factor = 0.95 
# iterations = 2000  


dimensions = 2
candidates = 2
lower_bound, upper_bound = -10, 10
reduction_factor = 0.97
iterations = 1500


# Initialize candidate positions and bounds
bounds = np.array([[lower_bound, upper_bound] for _ in range(dimensions)])
candidate_positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (candidates, dimensions))

fitness_history = []
best_fitness_history = []
best_position = None
best_fitness = float('inf')

for iteration in range(iterations):
    f_values = booth_function(candidate_positions)
    fitness_history.append(f_values)
    
    #  Update here -> Update best position
    current_best_idx = np.argmin(f_values)
    if f_values[current_best_idx] < best_fitness:
        best_fitness = f_values[current_best_idx]
        best_position = candidate_positions[current_best_idx].copy()
    best_fitness_history.append(best_fitness)
    probabilities = calculate_probabilities(f_values)
    
    new_positions = np.zeros_like(candidate_positions)
    for i in range(candidates):
        selected = roulette_wheel_selection(probabilities)
        new_positions[i] = candidate_positions[selected]
    
    bounds = update_bounds(bounds, reduction_factor, best_position)
    
    # Update here -> local search around best position
    candidate_positions = np.array([
        best_position + np.random.normal(0, (bounds[:, 1] - bounds[:, 0])/10, dimensions)
        for _ in range(candidates)
    ])
    candidate_positions = np.clip(candidate_positions, bounds[:, 0], bounds[:, 1])
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1}")
        print("Best Fitness:", best_fitness)
        print("Current Fitness Values:", f_values, "\n")
    if best_fitness < 1e-6:
        print(f"Converged to optimal solution at iteration {iteration + 1}")
        

# Convert fitness history to a numpy array for plotting
fitness_history = np.array(fitness_history)

# Plot convergence of fitness values
plt.figure(figsize=(10, 6))

# for i in range(candidates):
#     plt.plot(range(1, len(fitness_history) + 1), fitness_history[:, i], marker='o', label=f'Candidate {i + 1}')

# plt.xlabel('Iteration')
# plt.ylabel('f(X) (Fitness Values)')
# plt.hlines(y=1e-6, xmin=0, xmax=len(fitness_history))
# plt.title('Convergence of Booth Function Values Over Iterations')
# plt.grid(True)
# plt.legend()

# plt.savefig("booth_function_convergence.png")
# plt.show()



# Test check if _value_below_1e-6 

for i in range(candidates):
    mask = fitness_history[:, i] < 1e-6
    if np.any(mask):  
        iterations = np.arange(1, len(fitness_history) + 1)
        plt.plot(iterations[mask], 
                fitness_history[mask, i], 
                marker='o', 
                label=f'Candidate {i + 1}')

plt.hlines(y=1e-6, 
          xmin=0, 
          xmax=len(fitness_history), 
          colors='r', 
          linestyles='--', 
          label='Threshold (1e-6)')

plt.xlabel('Iteration')
plt.ylabel('f(X) (Fitness Values)')
plt.title('Convergence of Booth Function Values Below 1e-6')
plt.grid(True)
plt.legend()
plt.yscale('log')

total_below = np.sum(fitness_history < 1e-6)
plt.text(0.02, 0.98, 
         f'Total points below threshold: {total_below}', 
         transform=plt.gca().transAxes, 
         verticalalignment='top')
plt.show()

