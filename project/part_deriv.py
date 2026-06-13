
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


TASK_FOLDER = "part_results"
os.makedirs(TASK_FOLDER, exist_ok=True)


def compute_grad(m, n, p, n_grads, n_iter):
    sat_generator = kSATGenerator(k=2, seed=None)
    backend = AerSimulator()
    estimator = EstimatorV2()
    filename = f"{TASK_FOLDER}/m{m}_n{n}_p{p}.csv"
    if os.path.exists(filename):
        existing_rows = len(pd.read_csv(filename))  
    else:
        existing_rows = 0
    remaining = max(0, n_iter - existing_rows)
    for _ in range(remaining):
        formula = sat_generator.generate(m, n)
        sparse_pauli_list = sparse_pauli_list_ksat(formula)
        cost_hamiltonian = SparsePauliOp.from_list(sparse_pauli_list)
        
        raw_ansatz = QAOAAnsatz(cost_hamiltonian, reps=p)
        ansatz = transpile(raw_ansatz, backend=backend, optimization_level=1)

        def cost_function(params):
            pub = (ansatz, cost_hamiltonian, params)
            job = estimator.run([pub])
            result = job.result()[0]
            return result.data.evs

        partials = []
        for _ in range(n_grads):
            params = np.random.uniform(low=0, high=2*np.pi, size=ansatz.num_parameters)

            def f1(x):
                p_copy = params.copy()
                p_copy[0] = x[0]
                return cost_function(p_copy)

            partial = approx_fprime(np.array([params[0]]), f1)
            partials.append(partial)

        result= {
            "n_grads": n_grads,
            "mean_part": np.mean(partials),
            "std_part": np.std(partials),
            }
        pd.DataFrame([result]).to_csv(filename, index=False, mode="a", header=not os.path.exists(filename))


n_values=[10]
p_values=[1]
m_values=np.arange(2, 21, 2)

results = Parallel(n_jobs=-1)(
    delayed(compute_grad)(m, n, p, n_grads=5000, n_iter=50)
    for m in m_values
    for n in n_values
    for p in p_values
)
