import json
import os

def fill_latex_table(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 24 elements (4 models x 6 columns per model).
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    if len(values) != 26:
        raise ValueError("The values list must contain exactly 24 elements (4 models x 6 metrics each).")

    # Define the LaTeX table template with placeholders
    template = r"""
    \begin{{table}}[H]
    \centering
    \setlength\tabcolsep{{5pt}} % default value: 6pt
    \begin{{tabular}}{{lcccccc}}
    \hline
    \multicolumn{{1}}{{c}}{{\multirow{{2}}{{*}}{{\textbf{{Model}}}}}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Survival\\ Rate\end{{tabular}}}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Survival\\ Time\end{{tabular}}}} & \textbf{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Total\\ Gain\end{{tabular}}}} & \textbf{{Efficiency}} & \textbf{{Equality}} & \textbf{{Over-usage}} \\
    \multicolumn{{1}}{{c}}{{}}                                & Max = 100              & Max = 12               & Max = 120           & Max = 100           & Max = 1           & Min = 0             \\ \hline
    \multicolumn{{7}}{{l}}{{\textit{{\textbf{{Open-Weights Models}}}}}}                                                                                                                                   \\
    Llama-2-7b                                          & {0}                    & {1}                    & {2}                 & {3}                 & {4}               & {5}                 \\
    Llama-2-13b                                         & {6}                    & {7}                    & {8}                 & {9}                 & {10}              & {11}                \\
    Llama-3-8b                                          & {12}                   & {13}                   & {14}                & {15}                & {16}              & {17}                \\
    Mistral-7b                                          & {18}                   & {19}                   & {20}                & {21}                & {22}              & {23}                \\ \hline
    \end{{tabular}}
    \caption{{{24}}}
    \label{{tab:{25}}}
    \end{{table}}
    """

    # Format the template with the provided values
    filled_table = template.format(*values)
    return filled_table


def get_result_list(scenario, experiment):
    path = f"results_json/{scenario}/{experiment}"
    models = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Meta-Llama-3-8B-Instruct", "Mistral-7b-Instruct-v0.2"]
    if scenario == "fishing_v6.4":
        scen = "fish"
    elif scenario == "sheep_V6.4":
        scen = "sheep"
    else:
        scen  = "pollution"

    if experiment == "universalization":
        exp = "_universalization"
    else:
        exp = ""

    results = []
    for model in models:
        temp_results = {}
        model = f"{model}_{scen}_baseline_concurrent{exp}.json"
        with open(os.path.join(path, model), 'r') as file:
            temp_results = json.load(file)
            survival_rate = "{:.2f}".format(temp_results["general"]["survival_rate"]).replace("\\", "\\\\")
            survival_time = "{:.2f}".format(temp_results["general"]["mean_survival"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_survival'], 2))}}}"
            gains = "{:.2f}".format(temp_results["general"]["mean_gains"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_gains'], 2))}}}"
            efficiency = "{:.2f}".format(temp_results["general"]["mean_efficiency"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_efficiency'], 2))}}}"
            equality = "{:.2f}".format(temp_results["general"]["mean_equality"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_equality'], 2))}}}"
            usage = "{:.2f}".format(temp_results["general"]["mean_over_usage"]).replace("\\", "\\\\") + f"$\\pm$\\scriptsize{{{str(round(temp_results['general']['moe_confidence_over_usage'], 2))}}}"

        # results.extend([survival_rate, mean_survival, mean_gains, mean_efficiency, mean_equality, mean_over_usage])
        results.extend([survival_rate, survival_time, gains, efficiency, equality, usage])

    if scenario == "fishing_v6.4":
        scenario = "Fishing"
    elif scenario == "sheep_V6.4":
        scenario = "Pasture"
    else:
        scenario = "Pollution"

    if experiment == "universalization":
        experiment = "Universalization"
    else:
        experiment = "default"

    caption = f"Experiment: \\textit{{{experiment} - {scenario}}}"
    label = f"{scenario.lower()}_{experiment.lower()}"

    results.extend([caption, label])
    return results

if __name__ == '__main__':
    scenario = "fishing_v6.4"
    experiment = "baseline"

    scenarios = ["fishing_v6.4", "sheep_V6.4", "pollution_v6.4"]
    experiments = ["baseline", "universalization"]

    for scenario in scenarios:
        for experiment in experiments:
            # Get the results for the specified scenario and experiment
            results = get_result_list(scenario, experiment)
            table = fill_latex_table(results)

            with open(f"tables/{scenario}_{experiment}.tex", "w") as file:
                file.write(table)