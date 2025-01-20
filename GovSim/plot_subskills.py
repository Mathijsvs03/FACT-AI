import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind



def extract_subdirectory_names(path: str):
    """
    Function to extract the subdirectory names from a path.

    Args:
        path (str): Path to the directory.

    Returns:
        list: A list of subdirectory names.
    """
    subdirectory_names = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            subdirectory_names.append(dir_name)
        break # only get the first level directories
    return subdirectory_names


def extract_survival_time(path: str):
    """
    Function to extract survival times from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list: A list of survival times extracted from the JSON file.
    """
    with open(path, 'r') as f:
        data = json.load(f)

        for entrance, values in data.items():
            if entrance == "general":
                survival_time = values["mean_survival"]
    return survival_time

def extract_test_case_results(path: str):
    """
    Function to extract the test case results from the test json file

    Args:
        path: path to the test Json file.

    Returns:

    """

    mean_test_score = 0
    std_test_score = 0

    with open(path, 'r') as f:
        data = json.load(f)

        for entrance, values in data.items():
            if entrance == "score_mean":
                mean_test_score = values
            elif entrance == "score_std":
                std_test_score = values

    return mean_test_score, std_test_score


def plot_subskills(x, y, labels, title, sub_titles, x_label, y_label):
    """
    Function to plot the subskills results in a grid, with markers showing LLM performance.

    Args:
        x (dict): A dictionary containing the x values (test scores) for each subskill.
        y (dict): A dictionary containing the y values (survival times) for each subskill.
        labels (iterable): A list of labels for each LLM (e.g., model names).
        title (str): The title of the overall plot.
        sub_titles (list): A list of sub-titles for each subskill (test case names).
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
    """
    num_subskills = len(x)
    cols = 2
    rows = (num_subskills + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    markers = ['X', 's', 'D', '^']
    marker_sizes = [200, 125, 100, 100]

    for idx, (subskill_name, x_values) in enumerate(x.items()):
        ax = axes[idx]

        y_values = y[subskill_name]

        for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
            marker = markers[i % len(markers)]
            size = marker_sizes[i % len(marker_sizes)]
            ax.scatter(x_val, y_val, label=labels[i], marker=marker, s=size)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 12)
        ax.set_title(sub_titles[idx])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)

    for ax in axes[num_subskills:]:
        ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize='small', bbox_to_anchor=(0.5, 0.95))

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_path = os.path.join(plot_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()


def calculate_OLS_linear_regression(x, y):
    """
    Function to calculate the OLS linear regression for the given data.

    Args:
        x (list): A list of x values.
        y (list): A list of y values.

    Returns:
        float: R² value of the linear regression.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    return r_sq


def t_test(x, y):
    """
    Function to perform a t-test on the given data.

    Args:
        x (list): A list of x values.
        y (list): A list of y values.

    Returns:
        float: T-test value.
    """
    t_stat, p_value = ttest_ind(x, y)

    return t_stat, p_value


def main():
    parser = argparse.ArgumentParser(description="Process simulation and subskill paths.")
    parser.add_argument('--simulation_path', type=str, help='Path to the simulation results JSON files.', default='simulation/analysis_own/results_json')
    parser.add_argument('--subskill_path', type=str, help='Path to the subskill results.', default='subskills/results')
    parser.add_argument('--simulation_type', type=str, choices=['baseline', 'universalization'], help='Type of simulation for determining which survival time is used', default='baseline')

    args = parser.parse_args()
    simulation_path = args.simulation_path
    subskill_path = args.subskill_path
    simulation_type = args.simulation_type

    group_names = extract_subdirectory_names(simulation_path)

    # dictionary to store the model name and the survival time and has
    # the structure {group_name: {model_name: {experiment_type: survival_time_list}}}
    model_survival_time = {}

    # fill the dictionary
    for group_name in group_names:
        experiment_types = extract_subdirectory_names(os.path.join(simulation_path, group_name))
        model_survival_time[group_name.split('_')[0]] = {}
        for experiment_type in experiment_types:
            new_path = os.path.join(simulation_path, group_name, experiment_type)
            for root, dirs, files in os.walk(new_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(new_path, file)
                        model_name = file.split('_')[0]
                        if model_name not in model_survival_time[group_name.split('_')[0]]:
                            model_survival_time[group_name.split('_')[0]][model_name] = {}
                        if experiment_type not in model_survival_time[group_name.split('_')[0]][model_name]:
                            model_survival_time[group_name.split('_')[0]][model_name][experiment_type] = {}

                        survival_times = extract_survival_time(file_path)
                        model_survival_time[group_name.split('_')[0]][model_name][experiment_type] = survival_times

    test_file_2_test_case = {
        # 'multiple_math_consequence_after_fishing_same_amount.json' : 'simulation dynamics',
        'multiple_sim_consequence_after_fishing_same_amount.json': 'simulation dynamics',
        'multiple_sim_catch_fish_standard_persona.json' : 'sustainable action',
        # 'multiple_sim_universalization_catch_fish.json': 'sustainable action',
        # 'multiple_math_shrinking_limit_assumption.json':'sustainability threshold (assumption)',
        'multiple_sim_shrinking_limit_assumption.json': 'sustainability threshold (assumption)',
        # 'multiple_math_shrinking_limit.json' : 'sustainability threshold (belief)',
        'multiple_sim_shrinking_limit.json' : 'sustainability threshold (belief)',
        # 'multiple_math_consequence_after_using_same_amount.json': 'simulation dynamics',
        'multiple_sim_consequence_after_using_same_amount.json': 'simulation dynamics',
        'multiple_sim_consume_grass_standard_persona.json' : 'sustainable action',
        # 'multiple_sim_universalization_consume_grass.json' : 'sustainable action',
    }

    # dictionary to store the model name and the survival time and has
    # the structure {group_name: {model_name {test_name: mean_test_score}}}
    model_test_case = {}

    model_names = extract_subdirectory_names(subskill_path)


    for model_name in model_names:
        group_name = model_name.split('_')[1]
        if group_name not in model_test_case:
            model_test_case[group_name] = {}
        model_test_case[group_name][model_name.split('_')[0]] = {}

        run_names = extract_subdirectory_names(os.path.join(subskill_path, model_name))

        # Calculate the average test score for each model and specific test case
        test_name_mean_score = {}
        for run_name in run_names:
            run_path = os.path.join(subskill_path, model_name, run_name)
            test_names = []
            for root, dirs, files in os.walk(run_path):
                for file in files:
                    if file.endswith('.json'):
                        test_names.append(file)

            for test_name in test_names:
                if test_name not in test_file_2_test_case:
                    continue
                test_case_name = test_file_2_test_case[test_name]
                test_case_path = os.path.join(run_path, test_name)
                mean_test_score, _ = extract_test_case_results(test_case_path)
                if test_case_name not in test_name_mean_score:
                    test_name_mean_score[test_case_name] = []
                test_name_mean_score[test_case_name].append(mean_test_score)

        for test_name, mean_scores in test_name_mean_score.items():
            model_test_case[group_name][model_name.split('_')[0]][test_name] = np.mean(mean_scores)

    # Plot the individual subskills
    for group_name, model_dict in model_test_case.items():
        x = {}
        y = {}
        sub_titles = []
        for model_name, test_case_dict in model_dict.items():
            # To do Maybe add the experiment type as input argument of this program
            for test_case_name, test_case_score in test_case_dict.items():
                if test_case_name not in x:
                    x[test_case_name] = []
                    y[test_case_name] = []

                x[test_case_name].append(test_case_score)
                y[test_case_name].append(model_survival_time[group_name][model_name][simulation_type])

                if test_case_name not in sub_titles:
                    sub_titles.append(test_case_name)

        for subskill_name, x_values in x.items():
            y_values = y[subskill_name]
            r_sq = calculate_OLS_linear_regression(x_values, y_values)
            t_test_val, p_val = t_test(x_values, y_values)
            print(f"R² value for group name {group_name} and subskill name {subskill_name}: {r_sq} with p_value: {p_val}")

        plot_subskills(x=x, y=y, labels=list(model_dict.keys()), title=f"Subskills for {group_name}", sub_titles=sub_titles, x_label="Test Case Score", y_label="Survival Time")

if __name__ == '__main__':
    main()


