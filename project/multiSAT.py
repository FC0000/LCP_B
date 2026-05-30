import os
import numpy as np
from SATpackage import gridsearch_QAOA_SATsolver
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if __name__ == "__main__":
    data, metadata = gridsearch_QAOA_SATsolver(
        k=2,                            # define the kSAT problem
        num_vars=24,                     # number of qubits/variables in the problem
        p_values=[1, 2, 4, 8],      # QAOA depths
        m_values=np.arange(6, 43, 6),   # number of clauses in the kSAT problem
        n_trials=100,                    # number of random instances to test for each (p, m) pair
        N_samples=50000,                # number of samples to draw from the QAOA state for each instance
        optimizer="COBYLA",             # classical optimizer to use for tuning the QAOA parameters
        steps_optim=1500,               # number of optimization steps
        n_jobs=-1,                      # use all your CPU threads
        verbose=False,                   # print progress and results to the console
        dir_name="results_2sat",        # directory to save the results
        run_name="test_run4"            # name for this run to save the results under
    )