import abc
import os.path

import git
import numpy as np
import torch
from torch.utils.data import Dataset

from mpd.datasets.normalization import DatasetNormalizer
from mpd.utils.loading import load_params_from_yaml
from torch_robotics import environments, robots
from torch_robotics.environments import EnvDense2DExtraObjects
from torch_robotics.environments.env_simple_2d_extra_objects import EnvSimple2DExtraObjects
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

repo = git.Repo('.', search_parent_directories=True)
dataset_base_dir = os.path.join(repo.working_dir, 'data_trajectories')


# NOTE: dataset preprocessing
class TrajectoryDatasetBase(Dataset, abc.ABC):

    def __init__(self,
                 dataset_subdir=None,
                 include_velocity=False,
                 normalizer='LimitsNormalizer',
                 use_extra_objects=False,
                 obstacle_cutoff_margin=None,
                 tensor_args=None,
                 **kwargs):

        self.tensor_args = tensor_args

        self.dataset_subdir = dataset_subdir
        self.base_dir = os.path.join(dataset_base_dir, self.dataset_subdir)

        self.args = load_params_from_yaml(os.path.join(self.base_dir, '0', 'args.yaml'))
        self.metadata = load_params_from_yaml(os.path.join(self.base_dir, '0', 'metadata.yaml'))

        if obstacle_cutoff_margin is not None:
            self.args['obstacle_cutoff_margin'] = obstacle_cutoff_margin

        # -------------------------------- Load env, robot, task ---------------------------------
        # Environment
        env_class = getattr(
            environments, self.metadata['env_id'] + 'ExtraObjects' if use_extra_objects else self.metadata['env_id']) # NOTE: env comes from first folder in the dataset (data_trajectories, metadata.yaml)
        self.env = env_class(tensor_args=tensor_args)

        # Robot
        robot_class = getattr(robots, self.metadata['robot_id'])
        self.robot = robot_class(tensor_args=tensor_args)

        # Task
        self.task = PlanningTask(env=self.env, robot=self.robot, tensor_args=tensor_args, **self.args)
        self.planner_visualizer = PlanningVisualizer(task=self.task)

        # -------------------------------- Load trajectories ---------------------------------
        self.threshold_start_goal_pos = self.args['threshold_start_goal_pos']

        self.field_key_traj = 'traj'
        self.field_key_task = 'task'
        self.fields = {}

        # load data
        self.include_velocity = include_velocity
        self.map_task_id_to_trajectories_id = {}
        self.map_trajectory_id_to_task_id = {}
        self.load_trajectories()

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_traj].shape
        self.n_trajs = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.trajectory_dim = (self.n_support_points, d)

        # normalize the data (for the diffusion model)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        self.normalizer_keys = [self.field_key_traj, self.field_key_task]
        self.normalize_all_data(*self.normalizer_keys)

    def load_trajectories(self):
        # load free trajectories
        # print(f'$$$$$$$$$$$$$$$$$Loading trajectories from {self.base_dir}')
        trajs_free_l = []
        obstacles_l = []
        task_id = 0
        n_trajs = 0
        num_traj_in_dir = 0
        for current_dir, subdirs, files in os.walk(self.base_dir, topdown=True):
            if 'trajs-free.pt' in files:
                trajs_free_tmp = torch.load(
                    os.path.join(current_dir, 'trajs-free.pt'), map_location=self.tensor_args['device'])
                trajectories_idx = n_trajs + np.arange(len(trajs_free_tmp))
                self.map_task_id_to_trajectories_id[task_id] = trajectories_idx
                for j in trajectories_idx:
                    self.map_trajectory_id_to_task_id[j] = task_id
                task_id += 1
                n_trajs += len(trajs_free_tmp)
                num_traj_in_dir = len(trajs_free_tmp)
                trajs_free_l.append(trajs_free_tmp)
            if 'obstacles.pt' in files: # NOTE
                obstacles_tmp = torch.load(
                    os.path.join(current_dir, 'obstacles.pt'), map_location=self.tensor_args['device'])
                # print(f'$$$$$$$$$$$$$$$$$obstacles_tmp {obstacles_tmp.shape}') # torch.Size([num_traj_in_dir, cond_dim])
                obstacles_l.append(obstacles_tmp)
            
            # # load fixed obstacles params from env. TODO: add obstacles.pt to the dataset
            # if self.env.env_name == 'EnvSpheres3D':
            #     obstacles_tmp = self.env.spheres_flat
            #     obstacles_tmp = torch.unsqueeze(obstacles_tmp, 0).repeat(num_traj_in_dir, 1)
            #     obstacles_tmp += torch.randn_like(obstacles_tmp) * 0.01 # avoid normalization issues
            #     obstacles_l.append(obstacles_tmp)


        trajs_free = torch.cat(trajs_free_l)
        trajs_free_pos = self.robot.get_position(trajs_free)

        if self.include_velocity:
            trajs = trajs_free
        else:
            trajs = trajs_free_pos
        self.fields[self.field_key_traj] = trajs

        if len(obstacles_l) == 0:
            # task: start and goal state positions [n_trajectories, 2 * state_dim]
            task = torch.cat((trajs_free_pos[..., 0, :], trajs_free_pos[..., -1, :]), dim=-1) # NOTE: torch.Size([3600*4, 2*2]), position-only
            self.fields[self.field_key_task] = task
        else: # obstacle conditioning (EnvCage2D)
            obstacles = torch.cat(obstacles_l)
            self.fields[self.field_key_task] = obstacles

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f'{key}_normalized'] = self.normalizer(self.fields[f'{key}'], key)

    def render(self, task_id=3,
               render_joint_trajectories=False,
               render_robot_trajectories=False,
               **kwargs):
        # -------------------------------- Visualize ---------------------------------
        idxs = self.map_task_id_to_trajectories_id[task_id]
        pos_trajs = self.robot.get_position(self.fields[self.field_key_traj][idxs])
        start_state_pos = pos_trajs[0][0]
        goal_state_pos = pos_trajs[0][-1]

        fig1, axs1, fig2, axs2 = [None] * 4

        if render_joint_trajectories:
            fig1, axs1 = self.planner_visualizer.plot_joint_space_state_trajectories(
                trajs=pos_trajs,
                pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
                vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
            )

        if render_robot_trajectories:
            fig2, axs2 = self.planner_visualizer.render_robot_trajectories(
                trajs=pos_trajs, start_state=start_state_pos, goal_state=goal_state_pos,
            )

        return fig1, axs1, fig2, axs2

    def __repr__(self):
        msg = f'TrajectoryDataset\n' \
              f'n_trajs: {self.n_trajs}\n' \
              f'trajectory_dim: {self.trajectory_dim}\n'
        return msg

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        field_traj_normalized = f'{self.field_key_traj}_normalized'
        field_task_normalized = f'{self.field_key_task}_normalized'
        traj_normalized = self.fields[field_traj_normalized][index]
        task_normalized = self.fields[field_task_normalized][index]
        data = {
            field_traj_normalized: traj_normalized,
            field_task_normalized: task_normalized
        }

        # build hard conditions
        hard_conds = self.get_hard_conditions(traj_normalized, horizon=len(traj_normalized))
        data.update({'hard_conds': hard_conds})

        return data

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        raise NotImplementedError

    def get_unnormalized(self, index):
        raise NotImplementedError
        traj = self.fields[self.field_key_traj][index][..., :self.state_dim]
        task = self.fields[self.field_key_task][index]
        if not self.include_velocity:
            task = task[self.task_idxs]
        data = {self.field_key_traj: traj,
                self.field_key_task: task,
                }
        if self.variable_environment:
            data.update({self.field_key_env: self.fields[self.field_key_env][index]})

        # hard conditions
        # hard_conds = self.get_hard_conds(tasks)
        hard_conds = self.get_hard_conditions(traj)
        data.update({'hard_conds': hard_conds})

        return data

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize_trajectories(self, x):
        return self.unnormalize(x, self.field_key_traj)

    def normalize_trajectories(self, x):
        return self.normalize(x, self.field_key_traj)

    def unnormalize_tasks(self, x):
        return self.unnormalize(x, self.field_key_task)

    def normalize_tasks(self, x):
        return self.normalize(x, self.field_key_task)


class TrajectoryDataset(TrajectoryDatasetBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        # start and goal positions
        start_state_pos = self.robot.get_position(traj[0])
        goal_state_pos = self.robot.get_position(traj[-1])

        if self.include_velocity:
            # If velocities are part of the state, then set them to zero at the beggining and end of a trajectory
            start_state = torch.cat((start_state_pos, torch.zeros_like(start_state_pos)), dim=-1)
            goal_state = torch.cat((goal_state_pos, torch.zeros_like(goal_state_pos)), dim=-1)
        else:
            start_state = start_state_pos
            goal_state = goal_state_pos

        if normalize:
            start_state = self.normalizer.normalize(start_state, key=self.field_key_traj)
            goal_state = self.normalizer.normalize(goal_state, key=self.field_key_traj)

        if horizon is None:
            horizon = self.n_support_points
        hard_conds = {
            0: start_state,
            horizon - 1: goal_state
        }
        return hard_conds
