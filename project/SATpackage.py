import hashlib
import random
import os
import math
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from qiskit import transpile
from pysat.solvers import Glucose3
from scipy.sparse import diags
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_algorithms.optimizers import SPSA
from scipy.optimize import minimize
import json

# ----------------- Classical Algorithms for SAT generation and exact solving  ----------------- #

class kSATGenerator:
    """Generates random k-SAT formulas."""

    def __init__(self, k, seed=None):
        """
        Args:
            k (int): The number of literals per clause.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.k = k
        self.random = random.Random(seed)

    def generate(self, num_clauses, num_vars):
        """Generates a k-SAT formula with the specified number of clauses and variables.

        Args:
            num_clauses (int): The number of clauses to generate.
            num_vars (int): The total number of available variables.

        Returns:
            list: A list of tuples, where each tuple represents a clause of k literals.
        """
        # Check against the theoretical maximum of unique clauses
        max_num_clauses = 2**self.k * math.comb(num_vars, self.k)
        if num_clauses > max_num_clauses:
            raise ValueError("Too many clauses")
        
        vars_list = list(range(1, num_vars + 1))
        clauses = set()

        # Generate unique clauses using a set to automatically avoid duplicates
        while len(clauses) < num_clauses:
            variables = sorted(self.random.sample(vars_list, self.k))
            
            # Randomly assign positive or negative signs to the chosen variables
            literals = [var * self.random.choice([1, -1]) for var in variables]
            
            clause = tuple(literals)
            clauses.add(clause)

        return list(clauses)
    

class Pos1inkSATGenerator:
    """Generates random positive exactly-1-in-k SAT formulas."""

    def __init__(self, num_clauses, num_vars, k, seed=None):
        """
        Args:
            num_clauses (int): The number of clauses to generate.
            num_vars (int): The total number of available variables.
            k (int): The number of literals per clause.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.num_clauses = num_clauses
        self.num_vars = num_vars
        self.k = k
        self.random = random.Random(seed)

    def generate(self):
        """Generates the positive 1-in-k SAT formula.

        Returns:
            list: A list of tuples, where each tuple represents a clause of k positive literals.
        """
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
    """Evaluates whether a given SAT formula is satisfiable using the Glucose3 solver.

    Args:
        formula (list): A list of tuples representing the SAT formula.

    Returns:
        bool: True if the formula is satisfiable, False otherwise.
    """
    with Glucose3(bootstrap_with=formula) as solver:
        is_sat = solver.solve()
        
    return is_sat


def get_num_clauses(formula):
    """Returns the total number of clauses in a formula.

    Args:
        formula (list): The SAT formula.

    Returns:
        int: The number of clauses.
    """
    return len(formula)


def get_num_variables(formula):
    """Infers the number of unique variables in a formula.

    Args:
        formula (list): The SAT formula.

    Returns:
        int: The maximum variable index found in the formula.
    """
    return int(np.max(np.abs(np.array(formula).flatten())))


def get_k(formula):
    """Infers the value of k (literals per clause) from the first clause.

    Args:
        formula (list): The SAT formula.

    Returns:
        int: The length of the first clause.
    """
    return len(formula[0])


def brute_force_solve(formula, one_in_k=False):
    """Finds a satisfying bitstring for the formula by exhaustively searching all possible states.

    Args:
        formula (list): The SAT formula.
        one_in_k (bool, optional): If True, treats the formula as exactly-1-in-k SAT. Defaults to False.

    Returns:
        str or None: A string representing the satisfying bit assignment (e.g., "1011"), or None if no solution exists.
    """
    num_variables = get_num_variables(formula)
    
    # Iterate through all 2^N possible logical states
    for n in range(2**num_variables):
        violated_count = 0
        
        for clause in formula:
            true_count = 0
            for literal in clause:
                var_index = abs(literal) - 1
                
                # Extract the boolean value using bitwise shifts
                value = (n >> var_index) & 1
                
                # Flip the bit if the literal is negated
                if literal < 0:
                    value = 1 - value
                    
                true_count += value
            
            # Check satisfaction based on the chosen SAT variant
            if one_in_k:
                if true_count != 1:
                    violated_count += 1
            else:
                if true_count == 0:
                    violated_count += 1
        
        # Return the first fully satisfying state formatted as a binary string
        if violated_count == 0:
            bitstring = format(n, f"0{num_variables}b")
            return bitstring
            
    return None


def count_violated_clauses(formula, bitstring, one_in_k=False):
    """Counts the total number of violated clauses in a formula given a specific bitstring assignment.

    Args:
        formula (list): The SAT formula.
        bitstring (str): A string of binary digits representing the variable assignments.
        one_in_k (bool, optional): If True, evaluates violations based on exact-1-in-k logic. Defaults to False.

    Returns:
        int: The number of clauses violated by the given bitstring.
    """
    violated_count = 0
    for clause in formula:
        true_literals_count = 0
        for literal in clause:
            var_index = abs(literal) - 1
            bit_value = int(bitstring[-(var_index + 1)])
            var_value = bit_value if literal > 0 else (1 - bit_value)
            true_literals_count += var_value

        if one_in_k:
            if true_literals_count != 1:
                violated_count += 1
        else:
            if true_literals_count == 0:
                violated_count += 1
    return violated_count


def exact_maxsat(formula, num_vars):
    """Computes the maximum number of satisfiable clauses (C_max) by exploring all states.

    Args:
        formula (list): The SAT formula.
        num_vars (int): The total number of variables in the formula.

    Returns:
        int: The maximum possible number of satisfied clauses for the given formula.
    """
    N = 2**num_vars
    
    # Vectorized array representing all possible 2^N logical states
    states = np.arange(N, dtype=np.int32)
    violated_counts = np.zeros(N, dtype=np.int32)

    for clause in formula:

        clause_violated = np.ones(N, dtype=bool) # True for all states initially
        
        for lit in clause:
            var_idx = abs(lit) - 1
            
            # Extract the boolean value of this specific variable
            bit_values = (states >> var_idx) & 1
            
            if lit > 0:
                lit_false = (bit_values == 0)
            else:
                lit_false = (bit_values == 1)
                
            # The clause remains violated only for states where this current literal is also False
            clause_violated &= lit_false
            
        violated_counts += clause_violated
        
    min_violated = np.min(violated_counts)
    return len(formula) - min_violated



# ----------------- Quantum Algorithms for SAT solving with QAOA  ----------------- #


def ksat_hamiltonian(formula):
    """Generates the diagonal Hamiltonian matrix for a given k-SAT formula.

    Args:
        formula (list): A list of tuples representing the SAT formula.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix representing the diagonal cost Hamiltonian.
    """
    n_vars = get_num_variables(formula)

    # Total dimension of the Hilbert space (2^n states)
    dim = 2 ** n_vars
    H_diag = np.zeros(dim, dtype=int)

    for clause in formula:
        # A clause adds an energy penalty (1) only if all its literals evaluate to False
        ith_term = np.ones(dim, dtype=int)
        
        for literal in clause:
            variable_idx = np.abs(literal) - 1
            
            # Extract the boolean state of the current variable across all 2^n states simultaneously
            states = np.arange(dim)
            bits = (states >> variable_idx) & 1
            
            # Projector: evaluates to 1 if the specific literal is False, 0 otherwise
            if literal > 0:
                proj = 1 - bits
            else:
                proj = bits
                
            # Multiply projectors: evaluates to 1 only if all literals in the clause are False
            ith_term *= proj
            
        # Accumulate the penalty for this clause
        H_diag += ith_term
        
    H = diags(H_diag, 0, format='csr')
    return H


def sparse_pauli_list_pos1in2sat(formula):
    """Converts a positive exactly-1-in-2 SAT formula into a list of Pauli strings and coefficients.

    Args:
        formula (list): The positive 1-in-2 SAT formula.

    Returns:
        list: A list of tuples containing the Pauli string and its corresponding coefficient.
    """
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
        assert sgn1 == 1 and sgn2 == 1
        var1, var2 = np.abs(lit1), np.abs(lit2)

        add_term([], 1)
        add_term([var1, var2], 1)

    return [(pauli, coeff / 2) for pauli, coeff in coeffs.items() if coeff != 0]


def sparse_pauli_list_2sat(formula):
    """Converts a standard 2-SAT formula into a list of Pauli strings and coefficients.

    Args:
        formula (list): The 2-SAT formula.

    Returns:
        list: A list of tuples containing the Pauli string and its corresponding coefficient.
    """
    assert get_k(formula) == 2
    n_vars = get_num_variables(formula)
    
    coeffs = {}

    def add_term(z_vars, coeff):
        # Construct Qiskit-compatible Pauli string (qubit 0 is on the far right)
        string = ['I'] * n_vars
        for x in z_vars:
            string[-x] = 'Z'
        string = "".join(string)
        coeffs[string] = coeffs.get(string, 0) + coeff

    for (lit1, lit2) in formula:
        sgn1, sgn2 = int(np.sign(lit1)), int(np.sign(lit2))
        var1, var2 = np.abs(lit1), np.abs(lit2)

        # Expand the algebraic penalty term: (1 + s1*Z1) * (1 + s2*Z2)
        add_term([], 1)
        add_term([var1], sgn1)
        add_term([var2], sgn2)
        add_term([var1, var2], sgn1 * sgn2)

    return [(pauli, coeff / 4) for pauli, coeff in coeffs.items() if coeff != 0]


def sparse_pauli_list_ksat(formula):
    """Converts an arbitrary k-SAT formula into a list of Pauli strings and coefficients.

    Args:
        formula (list): The k-SAT formula.

    Returns:
        list: A list of tuples containing the Pauli string and its corresponding coefficient.
    """
    k = get_k(formula)
    n_vars = get_num_variables(formula)

    coeffs = {}

    def add_term(z_vars, coeff):
        # Construct Qiskit-compatible Pauli string (qubit 0 is on the far right)
        string = ['I'] * n_vars
        for q in z_vars:
            string[-q] = 'Z'
        string = "".join(string)
        coeffs[string] = coeffs.get(string, 0) + coeff

    for clause in formula:
        # Iteratively compute the tensor product expansion: \prod (1 + s_i * Z_i)
        terms = [(1, [])]
        for lit in clause:
            sgn = int(np.sign(lit))
            var = abs(lit)
            new_terms = []
            for coeff, z_vars in terms:
                # Branch 1: Multiply by Identity (1)
                new_terms.append((coeff, z_vars))
                # Branch 2: Multiply by Pauli-Z (s_i * Z_i)
                new_terms.append((coeff * sgn, z_vars + [var]))
            terms = new_terms

        # Add to global coeffs
        for coeff, z_vars in terms:
            add_term(z_vars, coeff)

    # Scale by 1/2^k as per the standard k-SAT cost Hamiltonian formulation
    return [(pauli, coeff / 2 ** k) for pauli, coeff in coeffs.items() if coeff != 0]


def success_probability(counts, formula):
    """Calculates the empirical probability of measuring a satisfying bitstring.

    Args:
        counts (dict): Measurement counts from the quantum sampler.
        formula (list): The SAT formula.

    Returns:
        float: The probability (between 0.0 and 1.0) of sampling a valid solution.
    """
    total_shots = sum(counts.values())
    success_shots = sum(count for bitstring, count in counts.items() if count_violated_clauses(formula, bitstring) == 0)
    return success_shots / total_shots


def expand_qaoa_params(old_params, p_old, p_new):
    """Expands the optimal parameters from an arbitrary depth p_old to p_new.
    
    Respects Qiskit's parameter ordering: [beta_0..beta_p, gamma_0..gamma_p].
    Missing layers are initialized with a small uniform noise near zero.

    Args:
        old_params (np.ndarray): The optimized parameters from the previous depth.
        p_old (int): The previous QAOA depth.
        p_new (int): The new, larger QAOA depth.

    Returns:
        np.ndarray: An expanded parameter array of size 2 * p_new.
    """
    delta_p = p_new - p_old
    
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
    """Custom termination checker for the SPSA optimizer to implement early stopping."""

    def __init__(self, patience=25, tol=1e-3, min_steps=1500):
        """
        Args:
            patience (int, optional): The sliding window size to evaluate convergence. Defaults to 25.
            tol (float, optional): The tolerance for the change in cost function. Defaults to 1e-3.
            min_steps (int, optional): The minimum number of steps before early stopping can trigger. Defaults to 1500.
        """
        self.patience = patience
        self.tol = tol
        self.min_steps = min_steps
        self.history = []
        self.steps = 0

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        """Evaluates whether the optimization should stop.

        Args:
            nfev (int): Number of function evaluations.
            parameters (np.ndarray): Current parameter values.
            value (float): Current objective function value.
            stepsize (float): Current step size.
            accepted (bool): Whether the step was accepted.

        Returns:
            bool: True if the optimizer should stop, False otherwise.
        """
        self.steps += 1
        self.history.append(value)
        
        if len(self.history) > self.patience:
            self.history.pop(0) 
            if self.steps > self.min_steps:
                window_spread = max(self.history) - min(self.history)
                if window_spread < self.tol:
                    return True 
                
        return False


def QAOA_SATsolver(
        formula,
        p,
        optimizer="L-BFGS-B",
        steps_optim=1000,
        initial_params=None,
        seed=None,
        verbose=False,
        trial_idx=None
    ):
    """Builds and optimizes a QAOA circuit for a given SAT formula.

    How it works:
    1. Transforms the SAT formula into a diagonal cost Hamiltonian (SparsePauliOp).
    2. Builds the parameterized QAOA ansatz circuit using Qiskit.
    3. Defines a cost function using EstimatorV2 to evaluate the expectation value of the Hamiltonian.
    4. Minimizes the cost function using the specified classical optimizer.

    Args:
        formula (list): The SAT formula to solve.
        p (int): The depth of the QAOA circuit.
        optimizer (str, optional): The classical optimizer to use ('COBYLA', 'L-BFGS-B', 'L-BFGS-B_grad', or 'SPSA'). Defaults to "L-BFGS-B".
        steps_optim (int, optional): Maximum number of iterations for the optimizer. Defaults to 1000.
        initial_params (np.ndarray, optional): Initial parameters for the circuit. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        verbose (bool, optional): Whether to print progress logs. Defaults to False.
        trial_idx (int, optional): The index of the current trial for tracking. Defaults to None.

    Returns:
        dict: A dictionary containing the optimization results (energy, execution time, optimal parameters).
    """
    # Initialize parameters randomly if no warm start is provided
    if initial_params is None:
        rng = np.random.default_rng(seed)
        initial_params = rng.uniform(0, 0.1, size=2*p)

    estimator = EstimatorV2()

    time_start = time.time()
    
    # Map the SAT formula to a quantum Ising Hamiltonian
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

    # Classical optimization block
    if optimizer == "L-BFGS-B_grad":
        # Use exact analytical gradient via parameter shift rule
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
            method="L-BFGS-B",
            jac=gradient_function,
            options={"maxiter": steps_optim}, 
        )
    
    elif optimizer == "L-BFGS-B":
        opt_result = minimize(
            cost_function, 
            initial_params, 
            method="L-BFGS-B",
            options={"maxiter": steps_optim}, 
        )

    elif optimizer == "SPSA":
        # SPSA with custom early stopping to prevent over-optimization
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
                
    energy_Hc = opt_result.fun

    time_end = time.time()
    delta_time = time_end - time_start

    m = get_num_clauses(formula)

    if verbose:
        print(f"(p={p}, m={m}), iter={trial_idx}: energy_Hc={energy_Hc:.4f}, time_sec={delta_time:.2f}")

    return {
        "optimizer": optimizer,
        "p": p,
        "m": m,
        "energy_Hc": energy_Hc,
        "time_sec": delta_time,
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
        steps_optim=1000,
        n_jobs=-1, 
        verbose=False,
        dir_name="results",
        run_name="default_run"
    ):
    """Executes a parallel grid search over different formula densities and QAOA depths.

    How it works:
    1. Distributes multiple trials across available CPU cores using Joblib.
    2. Inside each worker, generates a deterministic SAT formula based on a specific seed.
    3. Calculates the exact maximum number of satisfiable clauses (C_max) classically.
    4. Iterates through the sorted p_values (QAOA depths), optimizing the circuit for each depth 
       over the same formula in each CPU.
    5. Implements Heuristic Initialization: the optimal parameters from depth p are 
       expanded and passed as the initial starting point for depth p+delta_p, significantly 
       speeding up convergence and avoiding barren plateaus.
    6. Aggregates and saves the results into a CSV and metadata JSON.

    Note: 
        BEFORE RUNNING THIS FUNCTION REMEMBER TO SET:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        This prevents thread oversubscription between Scipy and Joblib.

    Args:
        k (int): Number of literals per clause (e.g., 2 or 3).
        num_vars (int): Number of variables (qubits).
        p_values (list): A list of QAOA depths to test. Must be sorted for proper heuristic initialization.
        m_values (list): A list of the number of clauses to test.
        n_trials (int, optional): Number of random formula instances per density. Defaults to 50.
        trial_start (int, optional): The starting index for the trial. Useful for resuming runs. Defaults to 0.
        optimizer (str, optional): The classical optimizer to use. Defaults to "L-BFGS-B".
        steps_optim (int, optional): Maximum optimization steps. Defaults to 1000.
        n_jobs (int, optional): Number of CPU cores to use for parallelization. Defaults to -1 (all).
        verbose (bool, optional): Print debug logs. Defaults to False.
        dir_name (str, optional): The folder where results will be saved. Defaults to "results".
        run_name (str, optional): The prefix name for the output files. Defaults to "default_run".

    Returns:
        tuple: A tuple containing the aggregated results DataFrame and the run metadata dictionary.
    """
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

    def worker(m, trial_idx):
        # Generate a deterministic seed based on clause density and trial index
        seed_string = f"m{m}_t{trial_idx}"
        trial_seed = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
        
        gen = kSATGenerator(k=k, seed=trial_seed)
        formula = gen.generate(num_clauses=m, num_vars=num_vars)
        c_max = exact_maxsat(formula, num_vars) 
        
        results_for_this_formula = []
        current_initial_params = None
        current_p = 0
        
        # Heuristic Initialization
        for p in sorted(p_values):
            if current_initial_params is None:
                rng = np.random.default_rng(trial_seed)
                current_initial_params = rng.uniform(0, 0.1, size=2*p)
            else:
                # Expand parameters to match the new depth p
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

            result["trial"] = trial_idx
            result["C_max"] = c_max
            result["C_qaoa"] = m - result["energy_Hc"]
            result["approx_ratio"] = result["C_qaoa"] / c_max if c_max > 0 else 1.0 

            results_for_this_formula.append(result)
            
        return results_for_this_formula

    total_tasks = len(m_values) * n_trials 
    print(f"Starting Grid Search: {total_tasks} unique formulas across {n_jobs if n_jobs > 0 else 'all'} cores...")
    time_start = time.time()

    # Distribute independent trials across CPU cores
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

    # Save outputs
    df_results = pd.DataFrame(results)
    csv_path = f"{dir_name}/{run_name}_n{num_vars}.csv"
    df_results.to_csv(csv_path, index=False)

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