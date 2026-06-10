import re
import pandas as pd
import hashlib
from SATpackage import kSATGenerator, exact_maxsat

# Parametri della simulazione originale
K = 2
NUM_VARS = 10
OPTIMIZER = "L-BFGS-B"
LOG_FILE = "log.csv"
OUTPUT_FILE = "reconstructed_results.csv"

def recover_metrics():
    results = []
    
    # Pattern per estrarre i dati dai log esattamente come stampati dal terminale
    log_pattern = re.compile(r"\(p=(\d+),\s*m=(\d+)\),\s*iter=(\d+):\s*energy_Hc=([0-9.-]+),\s*time_sec=([0-9.]+)")

    print(f"Reading {LOG_FILE} to recover missing metrics...")
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = log_pattern.search(line)
        if match:
            p = int(match.group(1))
            m = int(match.group(2))
            trial_idx = int(match.group(3))
            energy_Hc = float(match.group(4))
            time_sec = float(match.group(5))
            
            #Rigenera seed deterministico per questa iterazione
            seed_string = f"m{m}_t{trial_idx}"
            trial_seed = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
            
            #Rigenera formula logica
            gen = kSATGenerator(k=K, seed=trial_seed)
            formula = gen.generate(num_clauses=m, num_vars=NUM_VARS)
            
            #Ricalcola le massime clausole soddisfacibili (C_max)
            c_max = exact_maxsat(formula, NUM_VARS)
            
            #CalcolaC_qaoa e approximation ratio
            C_qaoa = m - energy_Hc
            approx_ratio = C_qaoa / c_max if c_max > 0 else 1.0
            
            #Dizionario come in SATpackage
            results.append({
                "optimizer": OPTIMIZER,
                "p": p,
                "m": m,
                "energy_Hc": energy_Hc,
                "time_sec": time_sec,
                "trial": trial_idx,
                "C_max": c_max,
                "C_qaoa": C_qaoa,
                "approx_ratio": approx_ratio
            })

    #DataFrame e salvataggio CSV
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"Done! {len(df_results)} valid iterations found.")
        print(f"Output saved in '{OUTPUT_FILE}'.\n")
        
        #Anteprima delle statistiche 
        summary_df = df_results.groupby(['p', 'm'])[['energy_Hc', 'approx_ratio', 'time_sec']].agg(['mean', 'std']).reset_index()
        summary_df.columns = ['_'.join(col).strip('_') if type(col) is tuple and col[1] else col[0] for col in summary_df.columns.values]
        print("Head of summary statistics:")
        print(summary_df.head())
    else:
        print("No valid data found.")

if __name__ == "__main__":
    recover_metrics()