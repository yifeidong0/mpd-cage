import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from mpd.models import build_context
from mpd.summaries.summary_base import SummaryBase


class SummaryTrajectoryGeneration(SummaryBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def summary_fn(self, train_step=None, model=None, datasubset=None, prefix='', debug=False, **kwargs):

        dataset = datasubset.dataset

        # ------------------------------------------------------------------------------------
        # Get a random task from the dataset
        trajectory_id = np.random.choice(datasubset.indices)
        task_id = dataset.map_trajectory_id_to_task_id[trajectory_id] # NOTE: task_id ~= trajectory_id // num_trajectories_per_task (num_envs)

        data_normalized = dataset[trajectory_id]
        # print(f"!!!!!!!!!!!!!!!!!!!!!data_normalized: {data_normalized.keys()}") # 'traj_normalized', 'task_normalized', 'hard_conds'
        context_normalized = build_context(model, dataset, data_normalized) # torch.Size([18])

        # ------------------------------------------------------------------------------------
        # Sample trajectories with the diffusion/cvae model
        n_samples = 25
        horizon = dataset.n_support_points
        hard_conds = data_normalized['hard_conds']

        trajs_normalized = model.run_inference(
            context_normalized, hard_conds,
            n_samples=n_samples, horizon=horizon,
            deterministic_steps=0
        )

        # Unnormalize trajectory samples from the diffusion model
        context = dict()
        trajs = dataset.unnormalize(trajs_normalized, dataset.field_key_traj) # torch.Size([n_samples, 64, 4])
        context['tasks'] = dataset.unnormalize(context_normalized['tasks'], dataset.field_key_task) # torch.Size([6, 3])

        # ------------------------------------------------------------------------------------
        # STATISTICS
        sphere_centers = context['tasks'].reshape(-1, 3)[:, :2]
        sphere_radii = context['tasks'].reshape(-1, 3)[:, 2]
        dataset.task.env.update_obstacles(sphere_centers, sphere_radii)

        wandb.log({f'{prefix}percentage free trajs': dataset.task.compute_fraction_free_trajs(trajs)}, step=train_step)
        wandb.log({f'{prefix}percentage collision intensity': dataset.task.compute_collision_intensity_trajs(trajs)}, step=train_step)
        wandb.log({f'{prefix}success': dataset.task.compute_success_free_trajs(trajs)}, step=train_step)

        # ------------------------------------------------------------------------------------
        # Render

        # dataset trajectory
        fig_joint_trajs_dataset, _, fig_robot_trajs_dataset, _ = dataset.render(
            task_id=task_id,
            render_joint_trajectories=True
        )

        # diffusion trajectory
        pos_trajs = dataset.robot.get_position(trajs)
        start_state_pos = pos_trajs[0][0]
        goal_state_pos = pos_trajs[0][-1]

        fig_joint_trajs_diffusion, _ = dataset.planner_visualizer.plot_joint_space_state_trajectories(
            trajs=pos_trajs,
            pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
            vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
            linestyle='dashed'
        )

        fig_robot_trajs_diffusion = None
        fig_robot_trajs_diffusion, _ = dataset.planner_visualizer.render_robot_trajectories(
            trajs=pos_trajs, start_state=start_state_pos, goal_state=goal_state_pos,
            linestyle='dashed'
        )

        if fig_joint_trajs_dataset is not None:
            wandb.log({f"{prefix}joint trajectories DATASET": wandb.Image(fig_joint_trajs_dataset)}, step=train_step)
        if fig_robot_trajs_dataset is not None:
            wandb.log({f"{prefix}robot trajectories DATASET": wandb.Image(fig_robot_trajs_dataset)}, step=train_step)

        if fig_joint_trajs_diffusion is not None:
            wandb.log({f"{prefix}joint trajectories DIFFUSION": wandb.Image(fig_joint_trajs_diffusion)}, step=train_step)
        if fig_robot_trajs_diffusion is not None:
            wandb.log({f"{prefix}robot trajectories DIFFUSION": wandb.Image(fig_robot_trajs_diffusion)}, step=train_step)

        if debug:
            plt.show()

        if fig_joint_trajs_dataset is not None:
            plt.close(fig_joint_trajs_dataset)
        if fig_robot_trajs_dataset is not None:
            plt.close(fig_robot_trajs_dataset)
        if fig_joint_trajs_diffusion is not None:
            plt.close(fig_joint_trajs_diffusion)
        if fig_robot_trajs_diffusion is not None:
            plt.close(fig_robot_trajs_diffusion)

