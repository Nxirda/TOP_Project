import subprocess
import numpy as np
import optuna as optuna
import os
import pandas as pd
from typing import Tuple, List


# Ugly scriptin but who cares
BIN_PATH = os.path.abspath("../Release/top-stencil")
CONFIG_PATH = os.path.abspath("../bench_config.txt")
REFERENCE_PATH = os.path.abspath("../reference/result_500x500x500.txt")
OUTPUT_PATH = os.path.abspath("../Release/tt.txt")

#
MAX_LIM = 1024

#
class StencilResults:
    def __init__(self, dims: Tuple[int, int, int], results: pd.DataFrame, runtime: List[float]):
        self.dims = dims
        self.results = results
        self.runtime = runtime

    def __str__(self):
        return f"Stencil results:\n\tDimensions: {self.dims}\n\tResults:\n{self.results}\n\tRuntimes:\n{self.runtime}\n\n"


#
def retrieve_results_and_runtime(file_path: str) -> Tuple[Tuple[int, int, int], pd.DataFrame, List[float]]:
    raw_data = pd.read_csv(file_path, header=None, delim_whitespace=True)
    simdims = tuple(map(int, raw_data.iloc[0, -3:].values))
    results = raw_data.iloc[:, 0].values
    runtime = raw_data.iloc[:, 1].values    
    return StencilResults(simdims, results, runtime)


#
def compare(ref: StencilResults, res: StencilResults) -> None:
    # Verify dimensions
    if ref.dims != res.dims:
        raise ValueError("Reference and result dimensions do not match.")

    # Initialize a flag to track whether any differences were found
    any_difference = False
    # Initialize a list to store the positions of incorrect cells
    incorrect_cells = []
    # Compare each cell in the results
    for i in range(len(ref.results)):
            diff = abs(ref.results[i] - res.results[i])
            if diff > 1e-12:
                print(f"\x1b[1;33mwarning:\x1b[0m difference found at iteration {i + 1}: {diff:e}")
                any_difference = True
                incorrect_cells.append(i)

    # If any differences found, raise an error
    if any_difference:
        error_message = "Results diverge too much from reference. Incorrect iterations: "
        error_message += ", ".join([f"{cell + 1}" for cell in incorrect_cells])
        raise ValueError(error_message)

    # Compare runtimes
    avg_runtime_ref = np.mean(ref.runtime)
    avg_runtime_res = np.mean(res.runtime)
    acc = ((avg_runtime_ref / avg_runtime_res) - 1.0) * 100.0
    if avg_runtime_ref < avg_runtime_res:
        print(f"\x1b[31mReference is {-acc:.2f}% faster than result\x1b[0m")
        return avg_runtime_res
    elif avg_runtime_ref > avg_runtime_res:
        print(f"\x1b[32mResult is {acc:.2f}% faster than reference\x1b[0m")
        return avg_runtime_res
    else:
        print("Reference and result have the same average runtime")
        return avg_runtime_res

#
def is_power_of_two(x):
    return x != 0 and (x & (x - 1)) == 0

#
def objective(trial):
    # Define the parameter space as powers of two
    param1 = trial.suggest_int('Block_x', 1, MAX_LIM)
    param2 = trial.suggest_int('Block_y', 1, MAX_LIM)
    param3 = trial.suggest_int('Block_z', 1, MAX_LIM)

    squared_Block_X = param1 ** 2
    squared_Block_Y = param2 ** 2
    squared_Block_Z = param3 ** 2

    # Run the program with the chosen parameters
    command = f"{BIN_PATH} {CONFIG_PATH} {OUTPUT_PATH} {str(squared_Block_X)} {str(squared_Block_Y)} {str(squared_Block_Z)}"

    print(command)
    process = subprocess.Popen(command, shell=True, text=True)
    process.wait()
    
    reference = retrieve_results_and_runtime(REFERENCE_PATH)
    results = retrieve_results_and_runtime(OUTPUT_PATH)
    metric = compare(reference, results)
    return metric

#
def run_autotunning():
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
     # Save the study results to a .data file
    with open("study_results.data", "w") as f:
        f.write("Best value: {}\n".format(study.best_value))
        f.write("Best params:\n")
        for key, value in study.best_params.items():
            f.write("{}: {}\n".format(key, value))
        f.write("Trials:\n")
        for trial in study.trials:
            f.write("Value: {}\n".format(trial.value))
            f.write("Params:\n")
            for key, value in trial.params.items():
                f.write("  {}: {}\n".format(key, value))
            f.write("\n")

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value)) 


if __name__ == "__main__":
    run_autotunning()
