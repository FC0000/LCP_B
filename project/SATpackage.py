import hashlib
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
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_algorithms.optimizers import SPSA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
import hashlib
# from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# ----------------- Classical Algorithms for SAT generation and exact solving  ----------------- #

class kSATGenerator:
    def __init__(self, k, seed=None):
        self.k = k
        self.random = random.Random(seed)

    def generate(self, num_clauses, num_vars):
        max_num_clauses = 2**self.k * math.comb(num_vars, self.k)
        if num_clauses > max_num_clauses:
            raise ValueError("Too many clauses")
        
        vars_list = list(range(1, num_vars + 1))
        clauses = set()

        while len(clauses) < num_clauses:
            variables = sorted(self.random.sample(vars_list, self.k))
            literals = [var * self.random.choice([1, -1]) for var in variables]
            clause = tuple(literals)
            clauses.add(clause)

        return list(clauses)
    
class Pos1inkSATGenerator:
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
    
def SATsolver(formula):
    """Finds if a formula is satisfiable."""

    with Glucose3(bootstrap_with=formula) as solver:
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


def exact_maxsat(formula, num_vars):
    N = 2**num_vars
    states = np.arange(N, dtype=np.int32)
    violated_counts = np.zeros(N, dtype=np.int32)

    for clause in formula:
        clause_violated = np.ones(N, dtype=bool)
        for lit in clause:
            var_idx = abs(lit) - 1
            bit_values = (states >> var_idx) & 1
            if lit > 0:
                lit_false = (bit_values == 0)
            else:
                lit_false = (bit_values == 1)
            clause_violated &= lit_false
        violated_counts += clause_violated
        
    min_violated = np.min(violated_counts)
    return len(formula) - min_violated


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


def expand_qaoa_params(old_params, p_old, p_new):
    """
    Expands the optimal parameters from an arbitrary depth p_old to p_new.
    Respects Qiskit's parameter ordering: [beta_0..beta_p, gamma_0..gamma_p].
    """
    delta_p = p_new - p_old
    
    # Se per qualche motivo p_new non è maggiore, non facciamo nulla
    if delta_p <= 0:
        return old_params
        
    new_params = np.zeros(2 * p_new)
    
    # Extract old betas and gammas
    old_betas = old_params[:p_old]
    old_gammas = old_params[p_old:]
    
    # Generate delta_p new random parameters (small noise) for the new layers
    new_betas_padding = np.random.uniform(-0.01, 0.01, size=delta_p)
    new_gammas_padding = np.random.uniform(-0.01, 0.01, size=delta_p)
    
    # Append the new layers to the old ones
    new_betas = np.append(old_betas, new_betas_padding)
    new_gammas = np.append(old_gammas, new_gammas_padding)
    
    # Reassemble in Qiskit's required order
    new_params[:p_new] = new_betas
    new_params[p_new:] = new_gammas
    
    return new_params

class SPSA_EarlyStopping:
    def __init__(self, patience=25, tol=1e-3, min_steps=1500):
        self.patience = patience
        self.tol = tol
        self.min_steps = min_steps
        self.history = []
        self.steps = 0

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        self.steps += 1
        self.history.append(value)
        
        if len(self.history) > self.patience:
            self.history.pop(0) 
            if self.steps > self.min_steps:
                window_spread = max(self.history) - min(self.history)
                if window_spread < self.tol:
                    return True 
                
        return False


def QAOA_SATsolver(formula,
                   p,
                   optimizer = "COBYLA",
                   steps_optim = 1000,
                   initial_params = None,
                   seed = None,
                   verbose = False,
                   trial_idx = None
                   ):
    
    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 0.1, size=2*p)

    estimator = EstimatorV2()
    # sampler = SamplerV2()
    # sampler.options.default_shots = N_samples

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
    if optimizer == "L-BFGS-B":

        estimator_grad = ParamShiftEstimatorGradient(estimator=estimator)
        def gradient_function(params):
            job = estimator_grad.run(
                circuits=[ansatz], 
                observables=[cost_hamiltonian], 
                parameter_values=[params]
            )
            result = job.result().gradients[0]
            return np.array(result).astype(float).flatten()
    
        opt_result = minimize(
            cost_function, 
            initial_params, 
            method=optimizer,
            jac=gradient_function,
            options={"maxiter": steps_optim}, 
        )

    elif optimizer == "SPSA":
        early_stopper = SPSA_EarlyStopping()
        spsa = SPSA(maxiter=steps_optim, termination_checker=early_stopper)
        opt_result = spsa.minimize(fun=cost_function, x0=initial_params)

    else:

        opt_result = minimize(
            cost_function, 
            initial_params, 
            method=optimizer,
            options={"maxiter": steps_optim}, 
        )
                
    # extract optimal parameters and sample from the optimal circuit
    optimal_circuit = ansatz.assign_parameters(opt_result.x)
    optimal_circuit.measure_all()
    # job = sampler.run([(optimal_circuit,)])
    # sample_result = job.result()[0]
    # counts = sample_result.data.meas.get_counts()

    energy_Hc = opt_result.fun
    # success_prob = success_probability(counts, formula)

    time_end = time.time()

    m = get_num_clauses(formula)

    if verbose:
        print(f"(p={p}, m={m}), iter={trial_idx}: energy_Hc={energy_Hc:.4f}, time_sec={time_end - time_start:.2f}", flush=True)

    return {"optimizer": optimizer,
            "p": p,
            "m": m,
            "energy_Hc": energy_Hc,
            "time_sec": time_end - time_start,
            "opt_params": opt_result.x.tolist() 
    }


def gridsearch_QAOA_SATsolver(
        k,
        num_vars,
        p_values, 
        m_values, 
        n_trials=50,
        trial_start=0,
        optimizer="L-BFGS-B",
        steps_optim=10000,
        n_jobs=-1, 
        verbose=False,
        dir_name="results",
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

    run_metadata = {
        "k": int(k),
        "num_vars": int(num_vars),
        "p_values": [int(p) for p in p_values],
        "m_values": [int(m) for m in m_values],
        "n_trials": int(n_trials),
        "trial_start": int(trial_start),
        "optimizer": str(optimizer),
        "steps_optim": int(steps_optim),
        "n_jobs": int(n_jobs),
        "run_name": str(run_name),
        "heuristic_initialization": True
    }

    # define the worker function that runs on each CPU core
    def worker(m, trial_idx):
        # create a seed for this specific run
        seed_string = f"m{m}_t{trial_idx}"
        trial_seed = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
        gen = kSATGenerator(k=k, seed=trial_seed)
        formula = gen.generate(num_clauses=m, num_vars=num_vars)
        c_max = exact_maxsat(formula, num_vars) 
        
        results_for_this_formula = []
        current_initial_params = None
        current_p = 0
        
        # ensure p_values are sorted for sequential layerwise training
        for p in sorted(p_values):
            if current_initial_params is None:
                rng = np.random.default_rng(trial_seed)
                current_initial_params = rng.uniform(0, 0.1, size=2*p)
            else:
                current_initial_params = expand_qaoa_params(current_initial_params, current_p, p)
            
            result = QAOA_SATsolver(
                formula=formula,
                p=p,
                optimizer=optimizer,
                steps_optim=steps_optim,
                initial_params=current_initial_params,
                seed=trial_seed,
                verbose=verbose,
                trial_idx=trial_idx
            )
            
            # Save the optimized parameters to initialize the next p iteration
            current_initial_params = np.array(result["opt_params"])
            current_p = p
            
            del result["opt_params"]

            # add metadata for tracking
            result["trial"] = trial_idx
            result["C_max"] = c_max
            result["C_qaoa"] = m - result["energy_Hc"]
            result["approx_ratio"] = result["C_qaoa"] / c_max if c_max > 0 else 1.0 

            results_for_this_formula.append(result)
            
        return results_for_this_formula

    # A task now only iterates over m and trials. The p loop is inside the worker.
    total_tasks = len(m_values) * n_trials 
    print(f"Starting Grid Search: {total_tasks} unique formulas across {n_jobs if n_jobs > 0 else 'all'} cores...")
    time_start = time.time()

    # parallelize the nested loops using joblib (con il nuovo trial_start)
    results_nested = Parallel(n_jobs=n_jobs)(
        delayed(worker)(m, t)
        for m in m_values 
        for t in range(trial_start, trial_start + n_trials)
    )
    
    # Flatten the list of lists returned by the workers
    results = [res for formula_results in results_nested for res in formula_results]

    execution_time = time.time() - time_start
    print(f"Executed all simulations in {execution_time:.2f} seconds.")
    run_metadata["total_execution_time_sec"] = execution_time

    os.makedirs(dir_name, exist_ok=True)

    # save data (CSV)
    df_results = pd.DataFrame(results)
    csv_path = f"{dir_name}/{run_name}_n{num_vars}.csv"
    df_results.to_csv(csv_path, index=False)

    # save metadata (JSON)
    json_path = f"{dir_name}/{run_name}_n{num_vars}_meta.json"
    with open(json_path, 'w') as json_file:
        json.dump(run_metadata, json_file, indent=4)

    summary_df = df_results.groupby(['p', 'm'])[['energy_Hc', 'approx_ratio', 'time_sec']].agg(['mean', 'std']).reset_index()
    summary_df.columns = ['_'.join(col).strip('_') if type(col) is tuple and col[1] else col[0] for col in summary_df.columns.values]

    if verbose:
        print(f"\nSaved results to: {csv_path}")
        print(f"Saved metadata to: {json_path}")
        print("\nHead of summary statistics:")
        print(summary_df.head())
    
    return summary_df, run_metadata