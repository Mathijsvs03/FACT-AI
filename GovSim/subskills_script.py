
import yaml
import random
import subprocess
import argparse


def generate_random_seeds(num_seeds):
    seed = 42
    ran_obj = random.Random(seed)
    seeds = [ran_obj.randint(0, 2 ** 32 - 1) for _ in range(num_seeds)]
    return seeds


def change_config(file_path, new_group_name, new_llm_path, new_seed):
    """
    Updates the `group_name`, `llm.path`, and `seed` variables in a YAML config file.

    Parameters:
        file_path (str): Path to the `config.yaml` file.
        new_group_name (str): New value for the `group_name` field.
        new_llm_path (str): New value for the `llm.path` field.
        new_seed (int): New value for the `seed` field.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        config['group_name'] = new_group_name
        config['llm']['path'] = new_llm_path
        config['seed'] = new_seed

        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

        print("Config updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_experiment(model, seed, experiment):
    command = f"python3 -m subskills.{experiment}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the experiment: {e} with model {model} and seed {seed}")
def main():
    parser = argparse.ArgumentParser(description="Run experiments with different models and configurations.")
    parser.add_argument('experiment', choices=['fishing', 'pollution', 'sheep'], help="The subskills experiment to run.")
    parser.add_argument('--num_seeds', type=int, default=5, help="The number of random seeds to generate.")
    args = parser.parse_args()

    models_model_path = {
        'Llama-2-7b-chat-hf': 'meta-llama/Llama-2-7b-chat-hf',
        'Llama-2-13b-chat-hf': 'meta-llama/Llama-2-13b-chat-hf',
        'Meta-Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'Mistral-7B-Instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2'
    }
    experiments_conf_path = {
        'fishing': 'subskills/fishing/conf/config.yaml',
        'pollution': 'subskills/pollution/conf/config.yaml',
        'sheep': 'subskills/sheep/conf/config.yaml'
    }

    experiment = args.experiment
    num_seeds = args.num_seeds
    config_path = experiments_conf_path[experiment]

    for model, llm_path in models_model_path.items():
        seeds = generate_random_seeds(num_seeds)
        for seed in seeds:
            group_name = f'{model}_{experiment}'
            change_config(config_path, group_name, llm_path, seed)
            run_experiment(model, seed, experiment)

if __name__ == "__main__":
    main()