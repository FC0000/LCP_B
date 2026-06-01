
from SATpackage import kSATGenerator, sparse_pauli_list_ksat
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from scipy.optimize import approx_fprime
from joblib import Parallel, delayed
import os
import pandas as pd


TASK_FOLDER = "gradient_results"
os.makedirs(TASK_FOLDER, exist_ok=True)

def get_result_path(m, n, p):
    return f"{TASK_FOLDER}/m{m}_n{n}_p{p}.csv"

def save_task_result(m, n, p, result):
    """Save a single result as its own CSV file with a unique name."""
    filename = get_result_path(m, n, p)
    pd.DataFrame([result]).to_csv(filename, index=False, mode="a", header=not os.path.exists(filename))
    return filename


def compute_grad(m, n, p, n_grads, n_iter):
    grad_means = []
    grad_stds = []
    sat2_generator = kSATGenerator(k=2, seed=None)
    backend = AerSimulator()
    estimator = EstimatorV2()
    for _ in range(n_iter):
        formula = sat2_generator.generate(m, n)
        sparse_pauli_list = sparse_pauli_list_ksat(formula)
        cost_hamiltonian = SparsePauliOp.from_list(sparse_pauli_list)
        
        raw_ansatz = QAOAAnsatz(cost_hamiltonian, reps=p)
        ansatz = transpile(raw_ansatz, backend=backend, optimization_level=1)

        def cost_function(params):
            pub = (ansatz, cost_hamiltonian, params)
            job = estimator.run([pub])
            result = job.result()[0]
            return result.data.evs

        grads = []
        for _ in range(n_grads):
            params = np.random.uniform(low=0, high=2*np.pi, size=ansatz.num_parameters)

            grad = approx_fprime(params, cost_function)
            grads.append(grad)
        grad_norms = np.linalg.norm(grads, axis=1)
        
        grad_means.append(np.mean(grad_norms))
        grad_stds.append(np.std(grad_norms))

    result= {
        "n_grads": n_grads,
        "n_iter": n_iter,
        "mean_mean_grad_norm": np.mean(grad_means),
        "mean_std_grad_norm": np.mean(grad_stds),
        }
    save_task_result(m, n, p, result)


n_values=[4]
p_values=[1, 2]
m_values=np.arange(2, 10, 2)

results = Parallel(n_jobs=-1)(
    delayed(compute_grad)(m, n, p, n_grads=1000, n_iter=5)
    for m in m_values
    for n in n_values
    for p in p_values
)

from pathlib import Path
for file in Path("gradient_results").glob("*.csv"):
    df = pd.read_csv(file)

    aggregated = (
        df.groupby("n_grads")
          .apply(
              lambda g: pd.Series({
                  "n_iter": g["n_iter"].sum(),
                  "mean_mean_grad_norm": (
                      g["mean_mean_grad_norm"] * g["n_iter"]
                  ).sum() / g["n_iter"].sum(),
                  "mean_std_grad_norm": (
                      g["mean_std_grad_norm"] * g["n_iter"]
                  ).sum() / g["n_iter"].sum(),
              })
          )
          .reset_index()
    )
    aggregated.to_csv(file, index=False)