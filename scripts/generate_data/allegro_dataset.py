import os
import time
import argparse
import yaml
import torch
import numpy as np
import pybullet as p
import pybullet_data
from torch_robotics.environments.primitives import MultiSphereField # added to avoid loop import
from torch_robotics.robots.robot_allegro import RobotAllegro

# Mapping of 16D robot joints to 20D PyBullet joints (skip fixed joints)
JOINT_MAP = [0, 1, 2, 3,  # Index Finger
             5, 6, 7, 8,  # Middle Finger
             10, 11, 12, 13,  # Ring Finger
             15, 16, 17, 18]  # T

# Only detect pairs involving fingertips and palm
COLLISION_DETECT_PAIRS = [(-1, 4), (-1, 9), (-1, 14), (-1, 19),  # Palm to fingertip
                          (4, 9), (4, 14), (9, 14),  # Fingertip to fingertip
                          (4, 19), (9, 19), (14, 19)]  # Fingertip to thumb
PARENT_CHILD_LINK_PAIRS = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4),  # Index Finger
                            (-1, 5), (5, 6), (6, 7), (7, 8), (8, 9),  # Middle Finger
                            (-1, 10), (10, 11), (11, 12), (12, 13), (13, 14),  # Ring Finger
                            (-1, 15), (15, 16), (16, 17), (17, 18), (18, 19)]  # Thumb Finger

def interpolate_trajectory(q_start, q_goal, num_steps=10):
    """
    Generate a linear interpolation between start and goal joint configurations.
    """
    trajectory = torch.linspace(0, 1, num_steps).unsqueeze(1) * (q_goal - q_start) + q_start
    return trajectory

def check_self_collision(robot_id, visualize=False):
    """
    Check if the robot is in self-collision.
    Returns True if a collision is detected.
    """
    # for i in [-1,]:
    # for i, j in COLLISION_DETECT_PAIRS:
    for i in range(-1, p.getNumJoints(robot_id)):
        for j in range(i+1, p.getNumJoints(robot_id)):
            if (i, j) in PARENT_CHILD_LINK_PAIRS:
                continue

            # contact_points = p.getContactPoints(robot_id, robot_id, i, j)
            contact_points = p.getClosestPoints(robot_id, robot_id, -0.01, i, j)
            if len(contact_points) > 0:
                print(f"    Joint pair ({i}, {j}) has {len(contact_points)} contact points with distance {contact_points[0][8]}")

                # Mark in red of the two links in contact
                if visualize:
                    for point in contact_points:
                        p.addUserDebugLine(point[5], point[6], [1, 0, 0], 5)
                    time.sleep(5)
                    p.removeAllUserDebugItems()
                    
                return True
    return False

def generate_trajectory(robot_allegro, allegro_id, num_envs=8, num_variations=128, visualize=False):
    """
    Generate and save trajectory datasets for the Allegro hand.
    """
    num_waypoints = 64
    max_failures = 5  # Maximum number of failures per variation

    t0 = time.time()
    variation_id = 0
    while variation_id < num_variations:
        print(f"\n# Variation {variation_id}")
        np.random.seed(variation_id)

        trajectories = []
        num_failures = 0
        q_start = torch.tensor(robot_allegro.jl_lower, dtype=torch.float32)  # Fixed start pose
        # q_start = torch.tensor(robot_allegro.jl_upper, dtype=torch.float32)  # Fixed start pose
        # print('jl_lower:', robot_allegro.jl_lower.shape)
        # print('jl_upper:', robot_allegro.jl_upper.shape)
        print(f"  Start: {q_start}")

        env_id = 0
        while env_id < num_envs:
            print(f"  # Environment {env_id}")
            
            # Random goal within joint limits (16D)
            q_goal = torch.rand(len(JOINT_MAP)) * (robot_allegro.jl_upper - robot_allegro.jl_lower) + robot_allegro.jl_lower
            # q_goal = torch.tensor(robot_allegro.jl_upper, dtype=torch.float32)  # Fixed start pose
            print(f"    Goal: {q_goal}")

            # Generate trajectory (16D)
            trajectory = interpolate_trajectory(q_start, q_goal, num_steps=num_waypoints)

            # Apply initial joint state (map 16D → 20D)
            for i, mapped_index in enumerate(JOINT_MAP):
                p.resetJointState(allegro_id, mapped_index, q_start[i].item())
                # print(f"joint {i} lower: {p.getJointInfo(allegro_id, mapped_index)[8]}")
                # print(f"joint {i} upper: {p.getJointInfo(allegro_id, mapped_index)[9]}")
                # print(f"number of joints: {p.getNumJoints(allegro_id)}")

            # Check for self-collision along the trajectory
            collision_detected = False
            for q in trajectory:
                # print(f"    Step {q}")
                # for i in range(p.getNumJoints(allegro_id)):
                #     print(f"joint {i} state: {p.getJointState(allegro_id, i)[0]}")

                for i, mapped_index in enumerate(JOINT_MAP):
                    p.setJointMotorControl2(allegro_id, mapped_index, p.POSITION_CONTROL, targetPosition=q[i].item())
                p.stepSimulation()
                time.sleep(1.0 / 60.0)  # Run at ~60Hz

                # Check for self-collision
                if check_self_collision(allegro_id):
                    print("    Self-collision detected! Retrying with a new goal.")
                    collision_detected = True
                    break

            if collision_detected:
                num_failures += 1
                if num_failures >= max_failures:
                    print("    Too many failures. Skipping this variation.")
                    break
                continue  # Retry with a new random goal

            # Store the trajectory
            trajectories.append(trajectory.numpy())
            env_id += 1

            # Visualization toggle
            if visualize:
                time.sleep(1)

        if num_failures >= max_failures:
            continue  # Skip this variation

        # Save dataset
        trajectories = np.array(trajectories)  # Shape: (num_envs, num_waypoints, num_allegro_dim)
        trajs_free = torch.tensor(trajectories, dtype=torch.float32)

        DATA_DIR = f"data_trajectories/EnvEmpty-RobotAllegro/{variation_id}"
        os.makedirs(DATA_DIR, exist_ok=True)
        torch.save(trajs_free, os.path.join(DATA_DIR, 'trajs-free.pt'))

        print(f"  Saved trajs-free.pt with shape: {trajs_free.shape}")

        # Save metadata
        args_save = {
            'debug': False,
            'device': 'cpu',
            'num_waypoints': num_waypoints,
            'num_trajectories': num_envs,
            'results_dir': DATA_DIR,
            'robot_id': 'RobotAllegro',
            'seed': variation_id
        }
        metadata = {
            'num_trajectories': num_envs,
            'num_trajectories_generated': num_envs,
            'num_trajectories_generated_coll': num_failures,
            'num_trajectories_generated_free': num_envs - num_failures,
            'robot_id': 'RobotAllegro'
        }

        with open(os.path.join(DATA_DIR, 'args.yaml'), 'w') as f:
            yaml.dump(args_save, f, default_flow_style=False)
        with open(os.path.join(DATA_DIR, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        print(f"  Metadata saved for variation {variation_id}")
        variation_id += 1

    print(f"Total time taken: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Allegro hand trajectory dataset.")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of trajectories per variation.")
    parser.add_argument("--num_variations", type=int, default=4, help="Total number of variations.")
    parser.add_argument("--visualize_pb", type=bool, default=0, help="Toggle PyBullet visualization.")
    args = parser.parse_args()

    # Initialize PyBullet
    physics_client = p.connect(p.GUI if args.visualize_pb else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load Allegro Hand URDF
    allegro_id = p.loadURDF(
        "/home/yif/Documents/KTH/git/mpd-cage/deps/torch_robotics/torch_robotics/data/urdf/robots/allegro_hand_description/allegro_hand_description_left.urdf",
        useFixedBase=True,
        globalScaling=10
    )

    # Initialize Allegro robot model
    tensor_args = {"device": "cpu", "dtype": torch.float32}
    robot_allegro = RobotAllegro(tensor_args=tensor_args)

    # Generate dataset
    generate_trajectory(robot_allegro, allegro_id, args.num_envs, args.num_variations, args.visualize_pb)

    # Disconnect PyBullet
    p.disconnect()
