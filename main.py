import numpy as np
from lle_algorithm import lle_epsilon_neighbors, lle_nearest_neighbors
from diffusion_map_algorithm import generate_diffusion_map, generate_diffusion_matrix
from visualization_algorithms import generate_2d_plot, generate_2d_plot_comparison, generate_3d_plot, generate_3d_plot_comparison, generate_digits_plot
from data_generation_algorithms import generate_unit_circle_points, generate_rotating_img_points, generate_unit_sphere_points, generate_3d_figure_eight, generate_3d_loop

#Setting Random Seed
np.random.seed(1)


# Main Parameters
n_points = 1000
n_dimension=2
#epsilon_list = [i/10 for i in range(1,11)]
epsilon_list = [.2]
# Method Type
# Options are 'diffusion_map' or 'lle_epsilon_neighbors', 'lle_nearest_neighbors
main_method = 'diffusion_map'


# LLE options
intrinsic_dimension = 2
if main_method == 'lle_nearest_neighbors':
    epsilon_list = [0]
# Number of nearest neighbors to find.
K = 20
dimension=2

# Points Parameters
# Options are "unit_circle", "unit_sphere", or "rotating_int", "loop", "figure_eight"
points_type = "figure_eight"
# Options are "uniform" or "beta" Distribution
distribution="beta"

# Noise Parameters
noise=False
noise_mean=0
noise_var=0.1

# Image Parameters
img_number = 13
n_images = 750
padding = 3

# Visualization Parameters
visualization_type = "2d"
if visualization_type == "2d" or visualization_type == "2d_comparison":
    dimension=2
elif visualization_type == "3d" or visualization_type == "3d_comparison":
    dimension=3
idx = [0,1]


for i in range(len(epsilon_list)):
    print("\n")
    print("Epsilon: " + str(epsilon_list[i]))
    print("Generating [" + points_type + "] Points...")

    if points_type == "unit_circle":
        points = generate_unit_circle_points(n_points,n_dimension, distribution=distribution,
                                              noise=noise, noise_mean=noise_mean, noise_var=noise_var)
    elif points_type == "unit_sphere":
        points = generate_unit_sphere_points(n_points,n_dimension,distribution=distribution,
                                              noise=noise,noise_mean=noise_mean,noise_var=noise_var)
    elif points_type == "rotating_int":
        points = generate_rotating_img_points(img_number, n_images, padding, noise=noise,                
                                               noise_mean=noise_mean,noise_var=noise_var)
    elif points_type == "loop":
        points = generate_3d_loop(n_points,distribution=distribution, 
                                  noise=noise, noise_mean=noise_mean, noise_var=noise_var)
    elif points_type == "figure_eight":
        points = generate_3d_figure_eight(n_points,distribution=distribution,
                                          noise=noise, noise_mean=noise_mean, noise_var=noise_var)
    else:
        print("Points Type [" + points_type + "] Not Found")
        break  

    # Each column in points is a data point.

    if main_method == 'diffusion_map':
        print("Generating Diffusion Matrix...")
        A, D_left = generate_diffusion_matrix(points, epsilon_list[i])

        print("Generating Diffusion Map...")
        results = generate_diffusion_map(A, D_left)

    elif main_method == 'lle_nearest_neighbors':
        results = lle_nearest_neighbors(points, dimension=dimension, K_neighbors=K)
    elif main_method == 'lle_epsilon_neighbors':
        results = lle_epsilon_neighbors(points, dimension=dimension, epsilon=epsilon_list[i])

    else:
        print("Method [" + main_method +"] Not Found")
        break

    print("Creating " + visualization_type + " Visualization...")

    print(results)
    

    if visualization_type == "2d":
        generate_2d_plot(results, idx=idx)
        generate_3d_plot(np.transpose(points), idx=idx)
    elif visualization_type == "2d_comparison":
        generate_2d_plot_comparison(results, points, idx=idx)
    elif visualization_type == "3d":
        generate_3d_plot(results,idx=idx)
    elif visualization_type == "3d_comparison":
        generate_3d_plot_comparison(results,idx=idx)
    elif visualization_type == "rotating_int":
        generate_digits_plot(results, points, idx=idx)
    else:
        print("Visualization Type [" + visualization_type + "] Not Found")
        break
