import glob
import json
import random
import warnings
from typing import Any, Dict, List, Optional

import sacred
import gymnasium
import numpy as np

from constraint_learning.util import logging
from constraint_learning.algos import cross_entropy
from constraint_learning.envs import controller_env, feature_wrapper

EnvConfig = Dict[str, Any]

ex = sacred.Experiment("highway_ce_test", ingredients=[])
ex.observers = [
    logging.SetID(),
    sacred.observers.FileStorageObserver("highway_ce_test"),
]
np.set_printoptions(suppress=True, precision=3)


def print_dict(info: Dict[str, Any], depth: str = 1, flush: bool = False):
    prefix = "\t" * depth
    for key, value in info.items():
        if isinstance(value, Dict):
            print(f"{prefix}{key:30}: {str(type(value)):>30}", flush=flush)
            print_dict(value, depth=depth + 1)
        elif isinstance(value, list) or isinstance(value, tuple):
            value = np.array(value)
            np.set_printoptions(precision=4, suppress=True)
            print(f"{prefix}{key:30}: {value}", flush=flush)
        elif isinstance(value, np.ndarray):
            np.set_printoptions(precision=4, suppress=True)
            if len(value.shape) > 1:
                print(f"{prefix}{key:30}: \n{value}", flush=flush)
            else:
                print(f"{prefix}{key:30}: {value}", flush=flush)

        elif isinstance(value, float):
            print(f"{prefix}{key:30}: {value:30.4f}", flush=flush)
        else:
            print(f"{prefix}{key:30}: {value:30}", flush=flush)

def make_env(
    env_name: str,
    env_config: EnvConfig,
    env_goal: controller_env.IntersectionGoal,
    reward_parameters: np.ndarray = np.zeros(9),
    constraint_parameters: Optional[np.ndarray] = None,
    constraint_thresholds: Optional[np.ndarray] = None,
    seed: int = 0,
):
    """Makes an environment from config and goal."""
    env = gymnasium.make(env_name,
                         render_mode=env_config['render_mode'] if 'render_mode' in env_config.keys() else None)
    env.configure(env_config)  # type: ignore
    env = controller_env.LinearVehicleIntersectionWrapper(env)
    env.set_goal(env_goal)  # type: ignore
    # We compute rewards and constraints directly from features, so here we can choose
    # a zero reward parameter.
    env = feature_wrapper.IntersectionFeatureWrapper(env,
                                                     reward_parameters=reward_parameters,
                                                     constraint_parameters=constraint_parameters,
                                                     constraint_thresholds=constraint_thresholds)
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    return env

def rollout_until_truncation(
    env: gymnasium.Env, max_duration: int, verbose: bool = False
) -> int:

    vehicle_params = controller_env.ControllerParameters(
        acceleration=[0.06653078952074894,
        0.6325689200679441,
        5.881895291905343],
        steering=[ 5.154084638597366,
        16.706544064918482],
    )
    env.set_parameters(vehicle_params)

    obs, info = env.reset()

    print(f"vehicle acc: {env.vehicle.ACCELERATION_PARAMETERS}, vehicle steer: {env.vehicle.STEERING_PARAMETERS}")
    print(f"vehicle acc range: \n{env.vehicle.ACCELERATION_RANGE}, \nvehicle steer range: \n{env.vehicle.STEERING_RANGE}")

    done, steps = False, 0
    total_reward = 0
    total_cost = 0
    features = 0
    while not done:
        action = env.ACTIONS_INDEXES["IDLE"] # actions depends on vehicle_params regardless of action
        # action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        total_reward += reward
        total_cost += info['cost']
        features += info['feature_vector']

        if verbose:
            print(f"feature: {info['feature_vector']}, cost: {info['cost']}")
            print(f"steps {steps}, speed: {env.vehicle.speed:.2f}, target_speed: {env.vehicle.target_speed:.2f}, {env.vehicle.action}, action: {action}")
        if steps >= max_duration or done:
            if verbose:
                print(f"\tEpisode reward: {total_reward/steps}, \tEpisode cost: {total_cost/steps}")
                print(f"\tEpisode Feature: {features/steps}")
                print(f"\tEpisode done at {steps} step")
                # print_dict(info)
            break
    return steps

@ex.named_config
def debug():
    verbose = True

@ex.config
def cfg():
    env_name = "Intersect-TruncateOnly-v0"
    allowed_goals = ["o1", "o2", "o3"]
    env_goal = "o1"
    reward_mean =  [0, 0, 0, 0.1, -0.2, 0, 0, 0, 0]
    reward_std =  [0, 0, 0, 0.1, 0.1, 0, 0, 0, 0]
    demonstration_folder = "demonstrations/"
    env_config = {
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "duration": 15,
        "render_mode": None, #'rgb_array',
    }
    num_thetas = 3
    save = False
    vebose = False

    iterations = 10
    num_candidates = 80
    num_elite = 10
    num_trajectories = 4
    num_jobs = 8
    solver_reinit = 1
    constraint_sort_method = "num_violations"

@ex.automain
def main(
    _run,
    env_name: str,
    allowed_goals: List[str],
    env_goal: str,
    reward_mean: List[float],
    reward_std: List[float],
    demonstration_folder: str,
    env_config: dict,
    num_thetas: int,
    iterations: int,
    num_candidates: int,
    num_elite: int,
    num_trajectories: int,
    num_jobs: int,
    solver_reinit: int,
    constraint_sort_method: str,
    save: bool = False,
    verbose: bool = False,
):
    constraint_parameters = np.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    constraint_thresholds = np.array(
        [
            0.2,  # speed_gt_limit
            0.2,  # too_close_to_front_vehicle
            0.05,  # collision
            0.1,  # not_on_street
        ]
    )

    seed = 0

    # Sample reward parameter from the specified distribution
    random.seed(seed)
    np.random.seed(seed)
    reward_mean, reward_std = np.array(reward_mean), np.array(reward_std)
    assert reward_mean.shape == reward_std.shape
    assert len(reward_mean.shape) == 1
    reward_parameters = reward_mean + np.random.randn(reward_mean.shape[0]) * reward_std
    assert env_goal in allowed_goals

    goal_idx = [
        controller_env.IntersectionGoal.LEFT,
        controller_env.IntersectionGoal.MIDDLE,
        controller_env.IntersectionGoal.RIGHT,
    ].index(env_goal)
    reward_parameters[goal_idx] = 10.0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*A Box observation space has an unconventional shape"
        )
        env = make_env(env_name, env_config, env_goal,
                       reward_parameters, constraint_parameters, constraint_thresholds, seed)
    print(f"reward params: {reward_parameters}:")

    rollout_until_truncation(env, 15, True)

    # solver = cross_entropy.CrossEntropySolver(
    #     env_name,
    #     env_config,
    #     num_jobs=num_jobs,
    #     constraint_sort_method=constraint_sort_method,
    # )
    # def callback(locals, globals):
    #     pass

    # for _ in range(num_thetas):
    #     obs, info = env.reset()
    #     done = False
    #     while not done:
    #         feature_vector = info['feature_vector']

    #         result = solver.solve(
    #         reward_parameters=reward_parameters,
    #         constraint_parameters=constraint_parameters,
    #         constraint_thresholds=constraint_thresholds,
    #         iterations=iterations,
    #         num_candidates=num_candidates,
    #         num_elite=num_elite,
    #         num_trajectories=num_trajectories,
    #         verbose=verbose,
    #         callback=callback,
    #         method="GT const",
    #         )

    #         print(f"\t{'feasible':30}: {result.feasible}")
    #         print(f"\t{'acceleration':30}: {result.acceleration}")
    #         print(f"\t{'steering':30}: {result.steering}")
    #         vehicle_params = controller_env.ControllerParameters(
    #             acceleration=result.acceleration,
    #             steering=result.steering,
    #         )
    #         env.set_parameters(vehicle_params)

    #         action = env.ACTIONS_INDEXES["IDLE"]

    #         obs, reward, done, truncated, info = env.step(action)

    #         cost = info['cost'] if 'cost' in info else 0.0
    #         print(f"\treward: {reward}, \tcost: {cost}")

    #         for key, value in info['features'].items():
    #             print(f"\t{key:30}: {value:10.2f}")

    #         if 'render_mode' in env_config.keys():
    #             env.render()

    #     for key, value in info.items():
    #         if key == 'features' or key == 'cost':
    #             continue
    #         print(f"\t{key:30}: {value:10.2f}")

    # def callback(locals, globals):
    #     pass

    # solver.solve(
    #         reward_parameters=reward_parameters,
    #         constraint_parameters=constraint_parameters,
    #         constraint_thresholds=constraint_thresholds,
    #         iterations=iterations,
    #         num_candidates=num_candidates,
    #         num_elite=num_elite,
    #         num_trajectories=num_trajectories,
    #         verbose=verbose,
    #         callback=callback,
    #     )