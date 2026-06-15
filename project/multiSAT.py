import os
from SATpackage import gridsearch_QAOA_SATsolver

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if __name__ == "__main__":
    data, metadata = gridsearch_QAOA_SATsolver(
        k=2,                                    # define the kSAT problem
        num_vars=10,                            # number of qubits/variables in the problem
        p_values=[1,2,4,8,16],                  # QAOA depths
        m_values=[2,4,6,8,10,12,14,16,18,20],   # number of clauses in the kSAT problem
        n_trials=25,                            # number of random instances to test for each (p, m) pair
        trial_start=0,                          # starting index for trial numbering (useful for resuming runs)
        optimizer="L-BFGS-B",                   # classical optimizer to use for tuning the QAOA parameters
        steps_optim=10000,                      # maximum number of optimization steps
        n_jobs=-1,                              # use all CPU threads
        verbose=True,                           # print progress and results to the console
        dir_name="results_2sat",                # directory to save the results
        run_name="run_2sat_bfgs_grad"           # name for this run to save the results under
    )