import glob
import json
import warnings
from typing import Any, Dict, List, Optional

import sacred
import numpy as np

from constraint_learning.util import logging
from constraint_learning.envs import controller_env
from constraint_learning.algos import cross_entropy
from constraint_learning.linear.highway_experiment import sample_demos


ex = sacred.Experiment("highway_ce_test", ingredients=[])
ex.observers = [
    logging.SetID(),
    sacred.observers.FileStorageObserver("highway_ce_test"),
]
np.set_printoptions(suppress=True, precision=3)


@ex.named_config
def debug():
    verbose = True

@ex.config
def cfg():
    demonstration_folder = "demonstrations/"
    num_thetas = 5
    env_name = "IntersectDefensive-TruncateOnly-v0"
    allowed_goals = ["left", "right"]
    env_config = {
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "duration": 15,
        "render_mode": 'rgb_array',
    }
    num_trajectories = 3
    vebose = False

@ex.automain
def main(
    _run,
    demonstration_folder: str,
    num_thetas: int,
    env_name: str,
    allowed_goals: List[str],
    env_config: dict,
    num_trajectories: int,
    verbose: bool = False,
):
    demonstration_files = glob.glob(f"{demonstration_folder}/*.json")
    phi, threshold = None, None

    all_demos = []
    for filename in demonstration_files:
        with open(filename, "r") as f:
            demo = json.load(f)

            if phi is None:
                phi = np.array(demo["constraint_parameters"])
                threshold = np.array(demo["constraint_thresholds"])
            assert np.allclose(phi, demo["constraint_parameters"]), (
                phi,
                demo["constraint_parameters"],
            )
            assert np.allclose(threshold, demo["constraint_thresholds"]), (
                threshold,
                demo["constraint_thresholds"],
            )

            features = np.array(demo["features"])
            if np.all(phi @ features <= threshold):
                all_demos.append(demo)
            else:
                if verbose:
                    print(f"Skipping demonstration {filename} because it is infeasible.")

    source_sample = sample_demos(
        all_demos=all_demos,
        num_samples=num_thetas,
        env_name=env_name,
        allowed_goals=allowed_goals,
    )

    print("="*100)
    for idx, demo in enumerate(source_sample):
        print(f"{idx} th demonstartion information:")
        for k, v in demo.items():
            print(f"\t{k}: {v}")

        # expert reward & constraint & policy
        reward_parameters = demo['reward_parameters']
        constraint_parameters = demo['constraint_parameters']
        optimal_features = demo['features']
        constraint_thresholds = demo['constraint_thresholds']
        optimal_reward = np.dot(optimal_features, reward_parameters)
        optimal_constraint = np.dot(constraint_parameters, optimal_features) - constraint_thresholds
        goal_param = np.array(demo['goal'])
        acceleration_param = np.array(demo['acceleration'])
        steering_param = np.array(demo['steering'])
        optimal_policy = np.concatenate([goal_param, acceleration_param, steering_param], axis=0)
        print(f"\toptimal_reward: {optimal_reward}")
        print(f"\toptimal_constraint: {optimal_constraint}")
        print(f"\toptimal_policy: {optimal_policy}")

        # implement to environment with demonsration information
        env_goal = cross_entropy.GOALS[goal_param.argmax()]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*A Box observation space has an unconventional shape"
            )
            env = cross_entropy._make_env(env_name, env_config, env_goal)

        vehicle_params = controller_env.ControllerParameters(
            acceleration=acceleration_param,
            steering=steering_param,
        )
        env.set_parameters(vehicle_params)

        def map_continuous_to_discrete(action_dict, acc_threshold=4.0, steer_threshold=0.3):
            acceleration = action_dict.get('acceleration', 0)
            steering = action_dict.get('steering', 0)

            # 속도 조절 액션 결정
            if acceleration > acc_threshold:
                speed_action = 'FASTER'
            elif acceleration < acc_threshold:
                speed_action = 'SLOWER'
            else:
                speed_action = 'IDLE'

            # 차선 변경 액션 결정
            if steering > steer_threshold:
                lane_action = 'LANE_RIGHT'
            elif steering < -steer_threshold:
                lane_action = 'LANE_LEFT'
            else:
                lane_action = 'IDLE'

            # 우선 순위 결정 (차선 변경 우선)
            if lane_action != 'IDLE':
                discrete_action = env.action_type.actions_indexes[lane_action]
            else:
                discrete_action = env.action_type.actions_indexes[speed_action]

            return discrete_action


        print("-"*100)
        for traj_idx in range(num_trajectories):
            print(f"\t{traj_idx} th trajectory information:")
            env.reset()

            done = False
            step = 0
            episode_length = env_config['duration'] * env_config['policy_frequency']
            while not done:
                # action = env.ACTIONS_INDEXES["IDLE"]

                # IDMVehicle의 액션 계산
                action_dict = env.vehicle.action
                acceleration = action_dict.get('acceleration')
                steering = action_dict.get('steering')
                action = map_continuous_to_discrete(action_dict)

                obs, reward, done, truncated, info = env.step(action)
                print(f"\t\taction: {action}, acc: {acceleration}, steer: {acceleration}, step : {step}")

                if env_config['render_mode'] in ['rgb_array', 'human'] if 'render_mode' in env_config.keys() else False:
                    env.render()
                    import time; time.sleep(0.1)
                    step += 1

            for k, v in info.items():
                print(f"\t\t{k}: {v}")

        print("="*100)