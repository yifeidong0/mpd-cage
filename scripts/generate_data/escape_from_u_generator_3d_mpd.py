"""
Title: 3D demonstration
Author: Your Name
Date: 2025-02-11
Description: The script demonstrates energy‐biased optimal path search in a 3D C‐space.
It is adapted from a 2D escape path demonstration by Yifei Dong.
This version uses a sphere robot (point–mass) and 10 spherical obstacles.
A potential objective is defined along the z-axis.
An option is provided to visualize the escape path in PyBullet.
"""
# NOTE: change to conda env of rob-tamp (also vscode interpreter)
import sys
import argparse
import matplotlib.pyplot as plt  # (optional: you can remove if not used)
import numpy as np
import torch
import os
import yaml
import time

# Import OMPL modules
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), '/home/yif/Downloads/omplapp-1.6.0-Source/ompl/py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

# Optionally import pybullet for visualization (only used when requested)
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None

# Global flags and hyperparameters
useGoalSpace = 0
useIncrementalCost = 1  # set True if need smooth paths
runtime = 1.0
planner = 'BITstar'  # 'BITstar'
runSingle = 0
gamma = 0.03
VISUALIZE = 0  # internal flag for other plots (if desired)
robot_radius = 0.15


########################################
# 3D State Validity and Potential Check
########################################

class ValidityChecker(ob.StateValidityChecker):
    def isValid(self, state):
        # For each obstacle, check that the clearance is positive.
        # valid_list = [self.clearance(state, radii[i], centers[i]) > 0.0 for i in range(len(radii))]
        # return all(valid_list)
    
        for i in range(len(radii)):
            if self.clearance(state, radii[i], centers[i]) <= 0.0:
                return False
        return True

    def clearance(self, state, radius, center):
        # In 3D, clearance = (distance from state to center)^2 - (radius)^2.
        x, y, z = state[0], state[1], state[2]
        xc, yc, zc = center[0], center[1], center[2]
        return (x - xc)**2 + (y - yc)**2 + (z - zc)**2 - (radius+robot_radius)**2

    
class minPathPotentialObjective(ob.OptimizationObjective):
    def __init__(self, si, start, useIncrementalCost):
        super(minPathPotentialObjective, self).__init__(si)
        self.si_ = si
        self.start_ = start
        self.useIncrementalCost_ = useIncrementalCost

    def combineCosts(self, c1, c2):
        if self.useIncrementalCost_:
            return ob.Cost(c1.value() + c2.value())
        else:
            return ob.Cost(max(c1.value(), c2.value()))

    def motionCost(self, s1, s2):
        # Use the z-axis as the potential (e.g. height)
        if self.useIncrementalCost_:
            return ob.Cost(abs(s2[2] - s1[2]))
        else:
            return ob.Cost(s2[2] - self.start_[2])

def getPotentialObjective(si, start, useIncrementalCost):
    return minPathPotentialObjective(si, start, useIncrementalCost)

def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj

def getBalancedObjective(si, start, useIncrementalCost):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    potentialObj = minPathPotentialObjective(si, start, useIncrementalCost)
    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, gamma)
    opt.addObjective(potentialObj, 1)
    return opt

def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)

def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj

def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bitstar":
        planner_obj = og.BITstar(si)
        planner_obj.params().setParam("rewire_factor", "0.5")
        planner_obj.params().setParam("samples_per_batch", "2000")
        return planner_obj
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def allocateObjective(si, objectiveType, start, useIncrementalCost):
    if objectiveType.lower() == "pathpotential":
        return getPotentialObjective(si, start, useIncrementalCost)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjWithCostToGo(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandpotential":
        return getBalancedObjective(si, start, useIncrementalCost)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")

def state_to_list(state):
    # Return a 3D list representation of the state.
    return [state[i] for i in range(3)]


########################################
# (Optional) 2D plotting functions
# (Left here for reference; 3D visualization is done with PyBullet.)
########################################

def plot_ellipse(center, radius, ax):
    # (This function is not used in 3D; it is kept here only as legacy code.)
    u = center[0]
    v = center[1]
    a = radius
    b = radius
    t = np.linspace(0, 2*np.pi, 100)
    x = u + a * np.cos(t)
    y = v + b * np.sin(t)
    ax.fill(x, y, alpha=0.7, color='#f4a63e')
    ax.plot(x, y, alpha=0.7, color='#f4a63e')


########################################
# PyBullet visualization function for 3D paths
########################################

def visualize_path_pybullet(sol_path, centers, radii, start_pos, goal_pos):
    if p is None:
        print("PyBullet is not installed. Cannot visualize.")
        return
    print("Visualizing path in PyBullet...")
    p.connect(p.GUI)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # planeId = p.loadURDF("plane.urdf")

    # Create obstacles as red spheres.
    obstacle_ids = []
    for center, r in zip(centers, radii):
        colId = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        visId = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1, 0, 0, 1])
        obs_id = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=colId,
                                   baseVisualShapeIndex=visId,
                                   basePosition=center)
        obstacle_ids.append(obs_id)

    # Create the robot as a small green sphere.
    colId = p.createCollisionShape(p.GEOM_SPHERE, radius=robot_radius)
    visId = p.createVisualShape(p.GEOM_SPHERE, radius=robot_radius, rgbaColor=[0, 1, 0, 1])
    robot_id = p.createMultiBody(baseMass=1,
                                 baseCollisionShapeIndex=colId,
                                 baseVisualShapeIndex=visId,
                                 basePosition=start_pos)

    # Mark the start and goal positions.
    p.addUserDebugText("Start", start_pos, textColorRGB=[0, 1, 0], lifeTime=0)
    p.addUserDebugText("Goal", goal_pos, textColorRGB=[1, 0, 0], lifeTime=0)

    # Draw the solution path as a blue line.
    if sol_path is not None:
        for i in range(len(sol_path) - 1):
            p.addUserDebugLine(sol_path[i], sol_path[i+1], [0, 0, 1], 2, lifeTime=0)
        #     p.addUserDebugLine([1, 1, 1], [-1, 1, 1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([-1, 1, 1], [-1, -1, 1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([-1, -1, 1], [1, -1, 1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([1, -1, 1], [1, 1, 1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([1, 1, -1], [-1, 1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([-1, 1, -1], [-1, -1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([-1, -1, -1], [1, -1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([1, -1, -1], [1, 1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([1, 1, 1], [1, 1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([-1, 1, 1], [-1, 1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([-1, -1, 1], [-1, -1, -1], [0, 1, 0], 2, lifeTime=0)
        #     p.addUserDebugLine([1, -1, 1], [1, -1, -1], [0, 1, 0], 2, lifeTime=0)
            time.sleep(0.05)

        # Animate the robot along the path.
        for pos in sol_path:
            p.resetBasePositionAndOrientation(robot_id, pos, [0, 0, 0, 1])
            p.stepSimulation()
            time.sleep(0.1)
        input("Press Enter to exit the PyBullet simulation...")

    p.disconnect()


########################################
# Planning function in 3D
########################################

def plan(runTime, plannerType, objectiveType, fname, centers, radii, start_pos, goal_pos, useIncrementalCost, visualize=0):
    # Create a 3D state space.
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(-1.0)
    bounds.setHigh(1.0)
    space.setBounds(bounds)

    # Construct the space information.
    si = ob.SpaceInformation(space)
    si.setup()

    # Set the validity checker.
    validityChecker = ValidityChecker(si)
    si.setStateValidityChecker(validityChecker)
    si.setup()

    # Create start and goal states.
    start = ob.State(space)
    start[0], start[1], start[2] = start_pos[0], start_pos[1], start_pos[2]
    goal = ob.State(space)
    goal[0], goal[1], goal[2] = goal_pos[0], goal_pos[1], goal_pos[2]

    # Potential (energy) is defined along the z-axis.
    Es, Eg = start[2], goal[2]

    # Create the problem definition.
    pdef = ob.ProblemDefinition(si)
    threshold = 0.001  # tolerance

    pdef.setStartAndGoalStates(start, goal, threshold)

    # Set the optimization objective.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType, start, useIncrementalCost))

    # Allocate and configure the planner.
    optimizingPlanner = allocatePlanner(si, plannerType)
    print("Planner parameters:", optimizingPlanner.params())
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # Attempt to solve the planning problem.
    solved = optimizingPlanner.solve(runTime)
    if solved:
        sol_path_geometric = pdef.getSolutionPath()
        objValue = sol_path_geometric.cost(pdef.getOptimizationObjective()).value()
        pathLength = sol_path_geometric.length()
        sol_path_states = sol_path_geometric.getStates()
        sol_path_list = [state_to_list(state) for state in sol_path_states]

        # Compute the potential cost as the maximum z reached minus the starting z.
        sol_path_z = [state[2] for state in sol_path_states]
        pathPotentialCost = max(sol_path_z) - start_pos[2]
        totalCost = gamma * pathLength + pathPotentialCost
        cost = totalCost

        print("pathPotentialCost:", pathPotentialCost)
        print("pathLengthCost:", pathLength)
        print('{0} found solution of path length {1:.4f} with an optimization objective value of {2:.4f}'.format(
            optimizingPlanner.getName(), pathLength, objValue))
        
        if fname:
            with open(fname, 'w') as outFile:
                outFile.write(sol_path_geometric.printAsMatrix())
    else:
        print("No solution found.")
        if visualize:
            print("No solution to visualize.")
        return None, None

    print('===================================')
    return pathPotentialCost, sol_path_list


########################################
# Downsampling a 3D path using cubic splines
########################################

from scipy.interpolate import CubicSpline
def downsample_path(path, cost, num_points=20):
    """
    Downsample a 3D path to `num_points` with smooth interpolation.
    
    Args:
        path (list of lists): Original path as a list of [x, y, z] points.
        num_points (int): Number of points in the downsampled path.
    
    Returns:
        np.ndarray: Downsampled smooth path with shape (num_points, 3).
    """
    path = np.array(path)  # shape (N, 3)
    # Parameterize by cumulative Euclidean distance.
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative_distances = np.hstack(([0], np.cumsum(distances)))
    
    # Create a cubic spline for each coordinate.
    cs_x = CubicSpline(cumulative_distances, path[:, 0])
    cs_y = CubicSpline(cumulative_distances, path[:, 1])
    cs_z = CubicSpline(cumulative_distances, path[:, 2])
    
    uniform_distances = np.linspace(0, cumulative_distances[-1], num_points)
    smooth_x = cs_x(uniform_distances)
    smooth_y = cs_y(uniform_distances)
    smooth_z = cs_z(uniform_distances)
    
    downsampled_path = np.vstack((smooth_x, smooth_y, smooth_z)).T
    return downsampled_path


########################################
# Random 3D obstacle and start/goal generator
########################################
def fibonacci_hemisphere(n, center, radius):
    """
    Generate n approximately uniformly distributed points over the lower half 
    (z < z_center) of a hemisphere using a Fibonacci spiral approach.

    Parameters:
        n (int): Number of points.
        center (tuple): Center of the hemisphere (x, y, z).
        radius (float): Radius of the hemisphere.

    Returns:
        np.ndarray: Array of shape (n, 3) containing the generated points.
    """
    x_c, y_c, z_c = center

    # Golden ratio for uniform distribution
    phi = (1 + np.sqrt(5)) / 2  
    
    # Generate Fibonacci spiral points on a sphere
    i = np.arange(1, n + 1)
    theta = 2 * np.pi * i / phi  # Spread points evenly in azimuth
    z = -1 + (2 * i - 1) / n  # Evenly spaced in z direction
    x = np.sqrt(1 - z**2) * np.cos(theta)
    y = np.sqrt(1 - z**2) * np.sin(theta)

    # Keep only lower hemisphere points (z < z_center)
    lower_mask = z < 0  
    x, y, z = x[lower_mask], y[lower_mask], z[lower_mask]

    # Scale by radius and shift to the given center
    x = x_c + radius * x
    y = y_c + radius * y
    z = z_c + radius * z

    return np.column_stack((x, y, z))

def generate_random_obstacles_3d(num_obstacles=6, fix_obstacles=False, fixed_centers=None, fixed_radii=None):
    """
    Generates a 3D environment with `num_obstacles` spherical obstacles arranged on the lower half
    of a spherical shell (i.e. with z < 0), forming a bowl-like obstacle space.
    The sphere object's start state is sampled from inside the bowl (with higher z values)
    and is guaranteed to be collision-free with respect to the obstacles.
    A fixed goal is set at [0, 0, 1] (representing an escape upward).

    Parameters:
        num_obstacles (int): Number of obstacles to generate.
        fix_obstacles (bool): If True, use the provided fixed_centers and fixed_radii.
        fixed_centers (array-like): Predefined obstacle centers if fix_obstacles is True.
        fixed_radii (array-like): Predefined obstacle radii if fix_obstacles is True.
    
    Returns:
        centers (np.ndarray): Array of shape (num_obstacles, 3) with obstacle centers.
        radii (np.ndarray): Array of shape (num_obstacles,) with obstacle radii.
        start_pos (list): A valid start position [x, y, z] (inside the bowl and collision-free).
        goal_pos (list): The fixed goal position [0, 0, 1].
    """
    import numpy as np

    # Define the spherical shell (bowl) parameters.
    base_shell_radius = 0.4          # Base radius of the shell.
    base_shell_center = np.array([0, 0, 0.5])  # Center of the shell; the bowl covers z <= 0.
    # thickness = 0.0                  # Variation in the radial distance.
    centers = fibonacci_hemisphere(num_obstacles*2, base_shell_center, base_shell_radius)
    centers += np.random.normal(0, 0.03, centers.shape)  # Add noise to the obstacle centers
    print(f"centers: {centers.shape}")

    if fix_obstacles and fixed_centers is not None and fixed_radii is not None:
        centers = np.array(fixed_centers)
        radii = np.array(fixed_radii)
    else:
        radii_list = []
        for _ in range(num_obstacles):
            obs_radius = np.random.uniform(0.15, 0.25)
            radii_list.append(obs_radius)
        # centers = np.array(centers_list)
        # print(f"centers: {centers.shape}")
        radii = np.array(radii_list)

    # Sample a valid start state for the sphere object.
    # We choose a region inside the bowl (with higher z values) so that the object is inside the bowl.
    # Here, we restrict theta to [0, pi/2] so that z = r*cos(theta) is nonnegative.
    while True:
        start_rad = np.random.uniform(0, base_shell_radius * 1.0)  # use a fraction of the shell radius
        start_theta = np.random.uniform(0, 2 * np.pi)              # only the upper half of the ball
        start_phi = np.random.uniform(0, 2 * np.pi)
        start_candidate = np.array([
            start_rad * np.sin(start_theta) * np.cos(start_phi) + base_shell_center[0],
            start_rad * np.sin(start_theta) * np.sin(start_phi) + base_shell_center[1],
            start_rad * np.cos(start_theta) + base_shell_center[2]
        ])
        # Ensure the start candidate is collision-free.
        collision = False
        for i in range(num_obstacles):
            if np.linalg.norm(start_candidate - centers[i]) < radii[i]+robot_radius:
                collision = True
                break
        if not collision:
            start_pos = start_candidate
            break

    # Set a fixed goal representing an escape
    goal_pos = np.array([0, 0, -1])
    return centers, radii, start_pos.tolist(), goal_pos.tolist()

########################################
# Dataset saving function
########################################

import joblib
def save_dataset(filename, dataset):
    """Save the dataset to a file."""
    joblib.dump(dataset, filename)
    print(f"Data has been saved to {filename}")
    loaded_data = joblib.load(filename)
    print("Loaded Data keys:", loaded_data.keys())


########################################
# Main
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D optimal motion planning demo program.')
    parser.add_argument('-t', '--runtime', type=float, default=runtime,
                        help='(Optional) Runtime in seconds (default: 1.0, > 0).')
    parser.add_argument('-p', '--planner', default=planner,
                        choices=['LBTRRT', 'BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar'],
                        help='(Optional) Specify the optimal planner to use (default: BITstar).')
    parser.add_argument('-o', '--objective', default='WeightedLengthAndPotential',
                        choices=['PathPotential', 'PathLength', 'ThresholdPathLength', 'WeightedLengthAndPotential'],
                        help='(Optional) Specify the optimization objective (default: WeightedLengthAndPotential).')
    parser.add_argument('-f', '--file', default=None,
                        help='(Optional) Specify an output file path for the found solution path.')
    parser.add_argument('-i', '--info', type=int, default=1, choices=[0, 1, 2],
                        help='(Optional) Set the OMPL log level: 0 for WARN, 1 for INFO, 2 for DEBUG.')
    parser.add_argument('--visualize_pb', type=bool, default=VISUALIZE,
                        help='(Optional) Visualize the escape path in PyBullet.')
    args = parser.parse_args()

    if args.runtime <= 0:
        raise argparse.ArgumentTypeError("Runtime must be greater than 0.")

    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")

    # Parameters for the dataset generation
    total_time = 5.0
    num_points = 64
    dt = total_time / num_points
    num_envs = 2       # Number of trajectories per variation
    num_variations = 4  # Total variations/environments
    t0 = time.time()

    # For the 3D case, we will use "EnvCage3D-RobotSphere" as the base directory.
    for variation_id in range(num_variations):
        print(f"\n# Variation {variation_id}")
        # Set a random seed for reproducibility
        np.random.seed(variation_id)
        
        costs = []
        paths = []
        velocities = []
        sphere_centers_list = []
        sphere_radii_list = []
        object_starts = []
        j = 0
        fixed_centers = None
        fixed_radii = None
        while j < num_envs:
            print(f"  # Environment {j}")
            try:
                # Generate random obstacles (10 spheres) and start/goal positions.
                centers, radii, start_pos, goal_pos = generate_random_obstacles_3d(fix_obstacles=False)
                # Optionally, you can fix the obstacles across environments by uncommenting:
                # if j == 0:
                #     fixed_centers = centers
                #     fixed_radii = radii
                # else:
                #     centers = fixed_centers
                #     radii = fixed_radii

                print("    centers:", centers)
                print("    radii:", radii)

                # Plan the escape path.
                pathPotentialCost, sol_path_list = plan(args.runtime, args.planner, args.objective, args.file,
                                                        centers, radii, start_pos, goal_pos, useIncrementalCost, visualize=VISUALIZE)
                if pathPotentialCost is not None:
                    sol_path = downsample_path(sol_path_list, pathPotentialCost, num_points=num_points)
                    
                    # Compute velocities from the way-points.
                    vel = np.diff(sol_path, axis=0) / dt  # (num_points-1, 3)
                    vel = np.vstack((vel, np.zeros((1, 3))))  # Append zero velocity at the end.
                    
                    costs.append(pathPotentialCost)
                    paths.append(sol_path)
                    velocities.append(vel)
                    object_starts.append(start_pos)
                    sphere_centers_list.append(centers)
                    sphere_radii_list.append(radii.tolist())
                    j += 1

                    # Visualize in PyBullet if the flag is set.
                    if args.visualize_pb:
                        visualize_path_pybullet(sol_path.tolist(), centers, radii, start_pos, goal_pos)
                else:
                    print("    No valid path found in this environment.")
            except Exception as e:
                print(f"    Error in environment {j}: {e}")
                continue  # Skip this environment if an error occurs

        # Save the trajectories dataset.
        # Trajectories: concatenation of positions and velocities along the last axis.
        trajectories = np.concatenate((np.array(paths), np.array(velocities)), axis=-1)
        trajs_free = torch.tensor(trajectories, dtype=torch.float32)
        DATA_DIR = f"EnvCage3D-RobotSphere/{variation_id}"
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        torch.save(trajs_free, os.path.join(DATA_DIR, 'trajs-free.pt'))
        print(f"trajs_free.shape: {trajs_free.shape}")  # Expected shape: (num_envs, num_points, 6)
        print(f"trajs_free.dtype: {trajs_free.dtype}")

        # Save obstacles: each obstacle is represented by [x, y, z, r].
        obstacles = np.hstack((centers, radii.reshape(-1, 1)))  # shape (10, 4)
        obstacles_flat = obstacles.flatten()
        obstacles_tile = np.tile(obstacles_flat, (num_envs, 1))
        obstacles_tensor = torch.tensor(obstacles_tile, dtype=torch.float32)
        print(f"obstacles.shape: {obstacles_tensor.shape}")
        torch.save(obstacles_tensor, os.path.join(DATA_DIR, 'obstacles.pt'))
        print("Time elapsed so far:", time.time() - t0)

        # Save YAML configuration files.
        args_save = {
            'debug': False,
            'device': 'cpu',
            'duration': total_time,
            'env_id': 'EnvCage3D',
            'git_hash': '9dd8739a99cd0a0ec1a690133b7dc71477082fc2',  # Update as needed.
            'git_url': 'git@github.com:yifeidong0/mpd-cage.git',
            'n_support_points': num_points,
            'num_trajectories': num_envs,
            'obstacle_cutoff_margin': 0.0,
            'results_dir': DATA_DIR,
            'robot_id': 'RobotSphere',
            'seed': variation_id,
            'threshold_start_goal_pos': 0.0
        }

        metadata = {
            'env_id': 'EnvCage3D',
            'num_trajectories': num_envs,
            'num_trajectories_generated': num_envs,
            'num_trajectories_generated_coll': 0,
            'num_trajectories_generated_free': num_envs,
            'robot_id': 'RobotSphere'
        }

        with open(os.path.join(DATA_DIR, 'args.yaml'), 'w') as f:
            yaml.dump(args_save, f, default_flow_style=False)
        with open(os.path.join(DATA_DIR, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

    print(f"Total time taken: {time.time() - t0:.2f}s")
