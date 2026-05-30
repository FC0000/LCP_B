import random
import os
import math
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from qiskit import transpile
from pysat.solvers import Glucose3
from scipy.sparse import diags
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from qiskit.primitives import StatevectorEstimator, StatevectorSampler


# ----------------- Classical Algorithms for SAT generation and exact solving  ----------------- #

class kSATGenerator:
    def __init__(self, num_clauses, num_vars, k, seed=None):
        self.num_clauses = num_clauses
        self.num_vars = num_vars
        self.k = k
        self.random = random.Random(seed)

    def generate(self):
        max_num_clauses = 2**self.k * math.comb(self.num_vars, self.k)
        if self.num_clauses > max_num_clauses:
            raise ValueError("Too many clauses")
        
        vars_list = list(range(1, self.num_vars + 1))
        clauses = set()

        while len(clauses) < self.num_clauses:
            variables = sorted(self.random.sample(vars_list, self.k))
            literals = [var * self.random.choice([1, -1]) for var in variables]
            clause = tuple(literals)
            clauses.add(clause)

        return list(clauses)


class Pos1in2SATGenerator:
    def __init__(self, num_clauses, num_vars, k, seed=None):
        self.num_clauses = num_clauses
        self.num_vars = num_vars
        self.k = k
        self.random = random.Random(seed)

    def generate(self):
        max_num_clauses = math.comb(self.num_vars, self.k)
        if self.num_clauses > max_num_clauses:
            raise ValueError("Too many clauses")
        
        vars_list = list(range(1, self.num_vars + 1))
        clauses = set()

        while len(clauses) < self.num_clauses:
            variables = sorted(self.random.sample(vars_list, self.k))
            literals = variables
            clause = tuple(literals)
            clauses.add(clause)

        return list(clauses)
    
def SATsolver(formula, seed):
    """Finds if a formula is satisfiable."""

    with Glucose3(shape=formula) as solver:
        is_sat = solver.solve()
        
    return is_sat

def get_num_clauses(formula):
    return len(formula)

def get_num_variables(formula):
    return int(np.max(np.abs(np.array(formula).flatten()))) # cast to int to avoid fixed size np.int64

def get_k(formula):
    return len(formula[0])

def brute_force_solve(formula, one_in_k=False):
    num_variables = get_num_variables(formula)
    for n in range(2**num_variables):
        #bitstring = [(n >> i) & 1 for i in reversed(range(num_variables))]
        violated_count = 0
        for clause in formula:
            true_count = 0
            for literal in clause:
                var_index = abs(literal) - 1
                value = (n >> var_index) & 1
                if literal < 0:
                    value = 1 - value
                    
                true_count += value
            if one_in_k:
                if true_count != 1:
                    violated_count += 1
            else:
                if true_count == 0:
                    violated_count += 1
        if violated_count == 0:
            bitstring = format(n, f"0{num_variables}b")
            return bitstring
    return None

def count_violated_clauses(formula, bitstring, one_in_k=False):
    violated_count = 0
    for clause in formula:
        true_literals_count = 0
        for literal in clause:
            var_index = abs(literal) - 1
            bit_value = int(bitstring[-(var_index + 1)])
            var_value = bit_value if literal > 0 else (1 - bit_value)
            true_literals_count += var_value

        if one_in_k:
            if true_literals_count != 1: # exactly one True needed
                violated_count += 1
        else:
            if true_literals_count == 0: # at least one True needed
                violated_count += 1
    return violated_count



# ----------------- Quantum Algorithms ----------------- #

def ksat_hamiltonian(formula):
    n_vars = get_num_variables(formula)

    dim = 2 ** n_vars
    H_diag = np.zeros(dim, dtype=int)

    for clause in formula:
        ith_term = np.ones(2 ** n_vars, dtype=int)
        for literal in clause:
            variable_idx = np.abs(literal) - 1
            
            states = np.arange(dim)
            bits = (states >> variable_idx) & 1
            if literal > 0:
                proj = 1 - bits
            else:
                proj = bits
            ith_term *= proj
        H_diag += ith_term
    H = diags(H_diag, 0, format='csr')
    return H


def sparse_pauli_list_pos1in2sat(formula):
    assert get_k(formula) == 2
    n_vars = get_num_variables(formula)
    
    coeffs = {}

    def add_term(z_vars, coeff):
        string = ['I'] * n_vars
        for x in z_vars:
            string[-x] = 'Z'
        string = "".join(string)
        coeffs[string] = coeffs.get(string, 0) + coeff

    for (lit1, lit2) in formula:
        sgn1, sgn2 = int(np.sign(lit1)), int(np.sign(lit2))
        assert sgn1==1 and sgn2==1
        var1, var2 = np.abs(lit1), np.abs(lit2)

        add_term([], 1)
        add_term([var1, var2], 1)

    return [(pauli, coeff / 2) for pauli, coeff in coeffs.items() if coeff != 0]


def sparse_pauli_list_2sat(formula):
    assert get_k(formula) == 2
    n_vars = get_num_variables(formula)
    
    coeffs = {}

    def add_term(z_vars, coeff):
        string = ['I'] * n_vars
        for x in z_vars:
            string[-x] = 'Z'
        string = "".join(string)
        coeffs[string] = coeffs.get(string, 0) + coeff

    for (lit1, lit2) in formula:
        sgn1, sgn2 = int(np.sign(lit1)), int(np.sign(lit2))
        var1, var2 = np.abs(lit1), np.abs(lit2)

        add_term([], 1)
        add_term([var1], sgn1)
        add_term([var2], sgn2)
        add_term([var1, var2], sgn1*sgn2)

    return [(pauli, coeff / 4) for pauli, coeff in coeffs.items() if coeff != 0]


def sparse_pauli_list_ksat(formula):
    k = get_k(formula)
    n_vars = get_num_variables(formula)

    coeffs = {}

    def add_term(z_vars, coeff):
        string = ['I'] * n_vars
        for q in z_vars:
            string[-q] = 'Z'
        string = "".join(string)
        coeffs[string] = coeffs.get(string, 0) + coeff

    for clause in formula:
        terms = [(1, [])]
        # Multiply by (1 + s_i Z_i)
        for lit in clause:
            sgn = int(np.sign(lit))
            var = abs(lit)
            new_terms = []
            for coeff, z_vars in terms:
                # Multiply by 1
                new_terms.append((coeff, z_vars))
                # Multiply by s_i Z_i
                new_terms.append((coeff * sgn, z_vars + [var]))
            terms = new_terms

        # Add to global coeffs
        for coeff, z_vars in terms:
            add_term(z_vars, coeff)

    return [(pauli, coeff / 2 ** k) for pauli, coeff in coeffs.items() if coeff != 0]


def success_probability(counts, formula):
    total_shots = sum(counts.values())
    success_shots = sum(count for bitstring, count in counts.items() if count_violated_clauses(formula, bitstring) == 0)
    return success_shots / total_shots


def QAOA_SATsolver(formula,
                   p,
                   N_samples = 10000,
                   optimizer = "Nelder-Mead",
                   initial_params = None,
                   seed = None,
                   verbose = False,
                   ):
    
    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 0.1, size=2*p)

    estimator = EstimatorV2()
    sampler = SamplerV2()
    sampler.options.default_shots = N_samples

    time_start = time.time()
                
    sparse_pauli_list = sparse_pauli_list_ksat(formula)
    cost_hamiltonian = SparsePauliOp.from_list(sparse_pauli_list)
                
    raw_ansatz = QAOAAnsatz(cost_hamiltonian, reps=p)
    backend = AerSimulator()
    ansatz = transpile(raw_ansatz, backend=backend, optimization_level=1)

    def cost_function(params):
        pub = (ansatz, cost_hamiltonian, params)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    # classical optimization with scipy
    opt_result = minimize(
        cost_function, 
        initial_params, 
        method=optimizer, 
        options={"maxiter":1000}, 
    )
                
    # extract optimal parameters and sample from the optimal circuit
    optimal_circuit = ansatz.assign_parameters(opt_result.x)
    optimal_circuit.measure_all()
    job = sampler.run([(optimal_circuit,)])
    sample_result = job.result()[0]
    counts = sample_result.data.meas.get_counts()

    energy_Hc = opt_result.fun
    success_prob = success_probability(counts, formula)

    time_end = time.time()

    m = get_num_clauses(formula)

    if verbose:
        print(f"(p={p}, m={m}): energy_Hc={energy_Hc:.4f}, success_prob={success_prob:.4f}, time_sec={time_end - time_start:.2f}")

    return {"optimizer": optimizer,
            "p": p,
            "m": m,
            "energy_Hc": energy_Hc,
            "success_prob": success_prob,
            "time_sec": time_end - time_start,
    }


def gridsearch_QAOA_SATsolver(
        k,
        num_vars,
        p_values, 
        m_values, 
        n_trials=10,
        N_samples=10000,
        optimizer="Nelder-Mead",
        n_jobs=-1, 
        verbose=False,
        dir_name="results_2sat",
        run_name="default_run"):
    
    '''
    
    BEFORE RUNNING THIS FUNCTION REMEMBER TO SET:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 

    => MUST be set before Qiskit/SciPy are called in parallel workers
       Indeed, scipy.optimize.minimize can use multiple threads internally, which can lead to oversubscription when combined with joblib's parallelism. 

    '''

    # define the worker function that runs on each CPU core
    def worker(p, m, trial_idx):
        # create a seed for this specific run
        trial_seed = hash(f"{run_name}_p{p}_m{m}_t{trial_idx}") % (2**32)
        
        # generate the formula using the TwoSAT generator we built earlier
        gen = kSATGenerator(num_clauses=m, num_vars=num_vars, k=k, seed=trial_seed)
        formula = gen.generate()
        
        # run the solver
        result = QAOA_SATsolver(
            formula=formula,
            p=p,
            seed=trial_seed,
            N_samples=N_samples,
            optimizer=optimizer,
            verbose=False
        )
        # add metadata for tracking
        result['trial'] = trial_idx
        return result

    total_tasks = len(p_values) * len(m_values) * n_trials
    print(f"Starting Grid Search: {total_tasks} total simulations across {n_jobs if n_jobs > 0 else 'all'} cores...")
    time_start = time.time()

    # parallelize the nested loops using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(p, m, t) 
        for p in p_values 
        for m in m_values 
        for t in range(n_trials)
    )

    print(f"Executed {total_tasks} simulations in {time.time() - time_start:.2f} seconds.")

    # data saving with pandas
    df_results = pd.DataFrame(results)
    os.makedirs(dir_name, exist_ok=True)
    df_results.to_csv(f"{dir_name}/{run_name}_n{num_vars}.csv", index=False)

    # group by 'p' and 'm', then calculate the mean and std across the n_trials
    summary_df = df_results.groupby(['p', 'm'])[['energy_Hc', 'success_prob', 'time_sec']].agg(['mean', 'std']).reset_index()
    summary_df.columns = ['_'.join(col).strip('_') if type(col) is tuple and col[1] else col[0] for col in summary_df.columns.values]

    if verbose:
        print("\nHead of summary statistics:")
        print(summary_df.head())
    
    return summary_df