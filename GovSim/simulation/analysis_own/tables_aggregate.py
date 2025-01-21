import json
import os
import numpy as np

def fill_latex_table(values):
    """
    Fill a LaTeX table template with values for a given model evaluation table.
    Args:
        values (list): A list of values to fill in the table.
                       The list must contain 26 elements (4 models x 6 columns + 2 for caption & label).
    Returns:
        str: A filled LaTeX table as a string.
    """
    # Ensure the values list has the correct number of elements
    if len(values) != 26:
        raise ValueError("The values list must contain exactly 26 elements (4 models x 6 metrics each + caption & label).")

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


if __name__ == '__main__':
    # We will aggregate the results over these scenarios and the single experiment "baseline".
    scenarios = ["fishing_v6.4", "sheep_V6.4", "pollution_v6.4"]
    experiments = ["baseline"]
    models = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Meta-Llama-3-8B-Instruct", "Mistral-7b-Instruct-v0.2"]

    # Dictionary to gather all metrics across scenarios for each model
    aggregator = {
        m: {
            "survival_rate": [],
            "mean_survival": [],
            "mean_gains": [],
            "mean_efficiency": [],
            "mean_equality": [],
            "mean_over_usage": []
        }
        for m in models
    }

    # Collect data from all scenarios (and experiments)
    for scenario in scenarios:
        for experiment in experiments:
            path = f"results_json/{scenario}/{experiment}"

            if scenario == "fishing_v6.4":
                scen = "fish"
            elif scenario == "sheep_V6.4":
                scen = "sheep"
            else:
                scen = "pollution"

            for model in models:
                model_long = f"{model}_{scen}_baseline_concurrent.json"
                with open(os.path.join(path, model_long), 'r') as file:
                    temp_results = json.load(file)
                    aggregator[model]["survival_rate"].append(temp_results["general"]["survival_rate"])
                    aggregator[model]["mean_survival"].append(temp_results["general"]["mean_survival"])
                    aggregator[model]["mean_gains"].append(temp_results["general"]["mean_gains"])
                    aggregator[model]["mean_efficiency"].append(temp_results["general"]["mean_efficiency"])
                    aggregator[model]["mean_equality"].append(temp_results["general"]["mean_equality"])
                    aggregator[model]["mean_over_usage"].append(temp_results["general"]["mean_over_usage"])

    # Now compute the average for each model across the 3 scenarios
    # We'll produce 6 columns per model (each as a formatted string).
    results_by_model = []

    for model in models:
        mean_survival_rate = np.mean(aggregator[model]["survival_rate"])
        mean_survival_time = np.mean(aggregator[model]["mean_survival"])
        mean_gain = np.mean(aggregator[model]["mean_gains"])
        mean_efficiency = np.mean(aggregator[model]["mean_efficiency"])
        mean_equality = np.mean(aggregator[model]["mean_equality"])
        mean_over_usage = np.mean(aggregator[model]["mean_over_usage"])

        # Convert each metric to a string with 2 decimals
        # (Adjust formatting as needed, e.g., ± confidence intervals, etc.)
        results_by_model.extend([
            f"{mean_survival_rate:.2f}",
            f"{mean_survival_time:.2f}",
            f"{mean_gain:.2f}",
            f"{mean_efficiency:.2f}",
            f"{mean_equality:.2f}",
            f"{mean_over_usage:.2f}"
        ])

    # Add caption and label for the aggregated table
    caption = "Aggregated baseline results (3 scenarios)"
    label = "aggregated_baseline"

    # Final list of 4×6 + 2 = 26 elements
    results_by_model.extend([caption, label])

    filled_table = fill_latex_table(results_by_model)
    print(filled_table)
