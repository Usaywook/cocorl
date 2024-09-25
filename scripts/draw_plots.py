import os
from collections import deque

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constraint_learning.util.plot_utils import plot_shadow_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="figs")
    parser.add_argument("--exp_label", type=str, default="gridworld")
    parser.add_argument("--study", type=str, default="no_transfer")
    parser.add_argument("--file", type=str, default="grid_aggregated.csv")
    parser.add_argument('--plot_keys', nargs='+', type=str, default=["safe_constraint_violation",
                                                                     "safe_reward"],
                        help='input plot keys')

    parser.add_argument('--adjust', nargs=4, type=float, default=[0.12, 0.96, 0.92, 0.12],
                        help='subplots_adjust(left=adjust[0], right=adjust[1], top=adjust[2], bottom=adjust[3])')
    parser.add_argument('--img_size', nargs=2, type=float, default=[6.4, 5.6],
                        help='figsize')
    parser.add_argument("--title_size", type=int, default=20)
    parser.add_argument("--axis_size", type=int, default=18)
    parser.add_argument("--legend_size", type=int, default=18)
    parser.add_argument("--linewidth", type=int, default=4)
    parser.add_argument("--line_alpha", type=float, default=0.8)
    parser.add_argument("--fill_alpha", type=float, default=0.2)

    parser.add_argument("--index_step", type=int, default=2)
    parser.add_argument("--average_num", type=int, default=2)
    parser.add_argument("--max_index", type=int, default=-1)
    parser.add_argument('--convention', action='store_true', help='Enable the conventional moving average (default: False)')
    parser.add_argument('--random_sampling', action='store_false', help='Disable the random sampling for moving average (default: True)')
    parser.add_argument('--norm_reward', action='store_false', help='Disable normalizing reward (default: True)')
    parser.add_argument('--save', action='store_true', help='Enable save plot (default: False)')

    return parser.parse_args()


class Plotter(object):
    def __init__(self, args):
        self._set_dir(args)
        self.precess_dict = self._process_df(args)

    def _set_dir(self, args):
        self.save_dir = os.path.join(args.save_dir, args.exp_label, args.study)
        os.makedirs(self.save_dir, exist_ok=True)

    def _set_parameters(self, args):
        self.plot_keys = args.plot_keys
        self.adjust = args.adjust
        self.img_size = args.img_size
        self.title_size = args.title_size
        self.axis_size = args.axis_size
        self.legend_size = args.legend_size
        self.linewidth = args.linewidth
        self.line_alpha = args.line_alpha
        self.fill_alpha = args.fill_alpha

        self.index_step = args.index_step
        self.average_num = args.average_num
        self.max_index = args.max_index

        self.random_sampling = args.random_sampling
        self.convention = args.convention
        self.norm_reward = args.norm_reward
        self.save = args.save

        self.exp_label = args.exp_label

        method_names_labels_dict_all = {
            "constraint_learning": "CoCoRL",
            "vanilla_irl_max_ent": "Average IRL",
            "known_reward_irl_max_ent": "Known IRL",
            "shared_reward_irl_max_ent": "Shared IRL",
        }
        self.method_names_labels_dict = {method: method_names_labels_dict_all[method] for method in self.unique_methods}

        label_dict_all = {
            "safe_constraint_violation": "Constraint Violation",
            "safe_reward": "Return",
        }
        self.label_dict = {key: label_dict_all[key] for key in self.plot_keys}

        plot_y_lim_dict_all = {
            "safe_constraint_violation": None, #(0,1),
            "safe_reward": None, #(0,8),
        }
        self.plot_y_lim_dict = {key: plot_y_lim_dict_all[key] for key in self.plot_keys}

        linestyle_dict_all = {"CoCoRL": "-",
                              "Average IRL": ":",
                              "Known IRL": "--",
                              "Shared IRL": "-."}
        self.linestyle_dict = {method: linestyle_dict_all[method_names_labels_dict_all[method]]
                               for method in self.unique_methods}

        colors_dict_all = {"CoCoRL": 'r',
                           "Average IRL": "b",
                           "Known IRL": "orange",
                           "Shared IRL": 'g'}
        self.colors = [colors_dict_all[method_names_labels_dict_all[method]]
                       for method in self.unique_methods]

    def _normalize_reward(self, df):
        reward_keys = [key for key in self.plot_keys if 'reward' in key]
        reward_keys = [key for key in df.keys() if key in reward_keys]
        for norm_key in reward_keys:
            max_value = abs(df[norm_key].max())
            df[norm_key] = df[norm_key].copy() / max_value

    def _process_df(self, args):
        df = pd.read_csv(args.file)
        self.unique_methods = df['method'].unique()

        self._set_parameters(args)

        if self.norm_reward:
            self._normalize_reward(df)

        all_mean_dict = {}
        all_std_dict = {}
        all_indice_dict = {}
        for method in self.unique_methods:
            method_df = df[df['method'] == method].copy()
            method_df = method_df.dropna(axis=1)
            seed_df_list = self._seed_df_list(method_df)

            all_results = []
            for seed_df in seed_df_list:
                results = {}
                for key in self.plot_keys:
                    results.update({key: list(seed_df[key].values)})
                all_results.append(results)

            mean_dict, std_dict, indice = self.mean_std_plot_results(all_results)

            all_mean_dict.update({method: {}})
            all_std_dict.update({method: {}})
            all_indice_dict.update({method: {}})
            for key in self.plot_keys:

                if self.average_num != 1:
                    mean_results_moving_average = self.compute_moving_average(result_all=mean_dict[key],
                                                                            average_num=self.average_num,
                                                                            random_sampling=self.random_sampling,
                                                                            convention=self.convention)
                    std_results_moving_average = self.compute_moving_average(result_all=std_dict[key],
                                                                            average_num=self.average_num,
                                                                            random_sampling=self.random_sampling,
                                                                            convention=self.convention)
                    indice_plot = indice[key][:len(mean_results_moving_average)]
                else:
                    mean_results_moving_average = mean_dict[key]
                    std_results_moving_average = std_dict[key]
                    indice_plot = indice[key]

                if self.max_index != -1:
                    mean_results_moving_average = mean_results_moving_average[:self.max_index]
                    std_results_moving_average = std_results_moving_average[:self.max_index]
                    indice_plot = indice_plot[:self.max_index]

                all_mean_dict[method].update({key: mean_results_moving_average})
                all_std_dict[method].update({key: std_results_moving_average})
                all_indice_dict[method].update({key: indice_plot})

        return {"mean": all_mean_dict,
                "std": all_std_dict,
                "indice": all_indice_dict}


    def compute_moving_average(self, result_all, average_num, random_sampling=False, convention=True):
        if convention:
            result_moving_average_all = []
            moving_values = deque([], maxlen=average_num)
            for result in result_all:
                moving_values.append(result)
                if len(moving_values) < average_num:  # this is to average the results in the beginning
                    result_moving_average_all.append(np.mean(result_all[:average_num]))
                else:
                    result_moving_average_all.append(np.mean(moving_values))
            return np.asarray(result_moving_average_all)
        else:
            if len(result_all) <= average_num:
                average_num = len(result_all)
            result_moving_all = []

            for i in range(average_num):
                if random_sampling:
                    filling_in_values = np.random.choice(result_all[-i:], i)
                else:
                    filling_in_values = result_all[len(result_all)-i:]

                result_moving_all.append(np.concatenate([result_all[i:], filling_in_values]))
            result_moving_all = np.mean(result_moving_all, axis=0)
            # return result_moving_all[:-average_num]
            return result_moving_all


    def _indice_step_fn(self, i):
        return max(i * self.index_step, 1)

    def _seed_df_list(self, df):
        df_list = []
        unique_seeds = df['seed'].unique()
        for seed in unique_seeds:
            seed_df = df[df['seed'] == seed].copy()

            seed_df.set_index("num_thetas", inplace=True)
            seed_df.sort_index(inplace=True)
            seed_df = seed_df[self.plot_keys]
            df_list.append(seed_df)
        return df_list

    def mean_std_plot_results(self, all_results):
        mean_results = {}
        std_results = {}
        indice = {}
        for key in all_results[0]:
            all_plot_values = []
            max_len = 0
            min_len = float('inf')
            for results in all_results:
                plot_values = results[key]
                if len(plot_values) > max_len:
                    max_len = len(plot_values)
                if len(plot_values) < min_len:
                    min_len = len(plot_values)
                all_plot_values.append(plot_values)

            plot_value_all = []
            for plot_values in all_plot_values:
                plot_value_all.append(plot_values[:min_len])
            for i in range(min_len, max_len):
                plot_value_t = []
                for plot_values in all_plot_values:
                    if len(plot_values) > i:
                        plot_value_t.append(plot_values[i])

                if 0 < len(plot_value_t) < len(all_plot_values):
                    for j in range(len(all_plot_values) - len(plot_value_t)):
                        plot_value_t.append(plot_value_t[j % len(plot_value_t)])  # filling in values
                for j in range(len(plot_value_t)):
                    plot_value_all[j].append(plot_value_t[j])

            mean_plot_values = np.mean(np.asarray(plot_value_all), axis=0)
            std_plot_values = np.std(np.asarray(plot_value_all), axis=0) / max(len(all_results) - 1, 1)
            mean_results.update({key: mean_plot_values})
            std_results.update({key: std_plot_values})
            indice.update({key: [i for i in range(max_len)]})

        return mean_results, std_results, indice


    def draw_plot(self):
        all_mean_dict = self.precess_dict["mean"]
        all_std_dict = self.precess_dict["std"]
        all_indice_dict = self.precess_dict["indice"]

        for key in self.plot_keys:
            mean_results_moving_avg_dict = {}
            std_results_moving_avg_dict = {}
            indice_dict = {}

            for method in self.unique_methods:
                mean_results_moving_avg_dict.update({method: all_mean_dict[method][key]})
                std_results_moving_avg_dict.update({method: all_std_dict[method][key]})
                indice_dict.update({method: all_indice_dict[method][key]})

            # title = self.exp_label.capitalize()
            title = self.label_dict[key]
            label = None # self.label_dict[key]
            save_name = os.path.join(self.save_dir, self.exp_label + '_' + key) if self.save else None

            self.plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                            std_results_moving_avg_dict=std_results_moving_avg_dict,
                            indice=indice_dict,
                            label=label,
                            method_names=self.unique_methods,
                            ylim=self.plot_y_lim_dict[key],
                            save_name=save_name,
                            legend_dict=self.method_names_labels_dict,
                            linestyle_dict=self.linestyle_dict,
                            title=title,
                            xlabel='Number of Demonstrations',
                            title_size=self.title_size,
                            legend_size=self.legend_size,
                            axis_size=self.axis_size,
                            img_size=self.img_size,
                            adjust=self.adjust,
                            linewidth=self.linewidth,
                            colors=self.colors,
                            )

    def plot_results(self, mean_results_moving_avg_dict,
                     std_results_moving_avg_dict,
                     indice,
                     ylim, label, method_names,
                     save_name=None,
                     xlabel='Episode',
                     legend_dict=None,
                     linestyle_dict=None,
                     adjust=None,
                     colors=None,
                     title_size=None,
                     legend_size=None,
                     axis_size=None,
                     img_size=None,
                     linewidth=None,
                     title=None):
        plot_mean_y_dict = {}
        plot_std_y_dict = {}
        plot_x_dict = {}
        for method_name in method_names:
            plot_x_dict.update({method_name: list(map(self._indice_step_fn, indice[method_name]))})
            plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
            plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})

        plot_shadow_curve(draw_keys=method_names,
                        x_dict_mean=plot_x_dict,
                        y_dict_mean=plot_mean_y_dict,
                        x_dict_std=plot_x_dict,
                        y_dict_std=plot_std_y_dict,
                        img_size=img_size if img_size is not None else (6, 5.8),
                        ylim=ylim,
                        title=title,
                        xlabel=xlabel,
                        ylabel=label,
                        legend_dict=legend_dict,
                        legend_size=legend_size if legend_size is not None else 18,
                        linestyle_dict=linestyle_dict,
                        axis_size=axis_size if axis_size is not None else 18,
                        adjust=adjust,
                        colors=colors,
                        line_alpha=self.line_alpha,
                        fill_alpha=self.fill_alpha,
                        linewidth=linewidth if linewidth is not None else 3,
                        title_size=title_size if axis_size is not None else 20,
                        save_name=save_name, )

def main():
    args = parse_args()
    print(args)

    plotter = Plotter(args)
    plotter.draw_plot()


if __name__ == "__main__":
    main()