import wandb
import json

api = wandb.Api(timeout=30)
runs = api.runs("EMS", filters={"tags": {"$nin": ["skip"]}})

errs = 0

for run in runs:
    try:
        run_name = run.name
        run_runtime = run.summary['_wandb']['runtime']
        run_created_at = run.createdAt

        experiment_name = run.config['experiment']["name"]
        scenario_name = run.config['experiment']["scenario"] + "_" + run.config['code_version']
        group_name = run.config["group_name"]

        try:
            univ = run.config['experiment']["env"]["inject_universalization"]
            if univ:
                reasoning_type = "universalization"
            else:
                reasoning_type = "baseline"

        except KeyError:
            reasoning_type = run.config['experiment']["env"]["inject_social_reasoning"]

        try:
            with open(f"results_json/{scenario_name}/{reasoning_type}/{group_name}.json", "r") as f:
                data = json.load(f)
                data[run_name]["runtime"] = run_runtime
                data[run_name]["created_at"] = run_created_at

            with open(f"results_json/{scenario_name}/{reasoning_type}/{group_name}.json", "w") as f:
                json.dump(data, f, indent=4)

        except FileNotFoundError:
            print(f"File error with {run.name} from {scenario_name}/{reasoning_type}/{group_name}")

    except KeyError:
        print(f"Error with {run.name}")
        errs += 1

print(f"Total errors: {errs}")