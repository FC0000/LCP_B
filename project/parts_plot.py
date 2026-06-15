import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TASK_FOLDER = "project/part_3sat_2026-06-15_18-35"
assert os.path.exists(TASK_FOLDER)

pattern = re.compile(r"m(\d+)_n(\d+)_p([0-9.eE+-]+)\.csv")

records = []

for fname in glob.glob(os.path.join(TASK_FOLDER, "m*_n*_p*.csv")):
    base = os.path.basename(fname)

    match = pattern.match(base)
    if match is None:
        continue

    m = int(match.group(1))
    n = int(match.group(2))
    assert n==10
    p = float(match.group(3))

    df = pd.read_csv(fname)
    n_rows = len(df)
    #if n_rows < 10:
    #    continue
    
    records.append({
        "m": m,
        "p": p,
        "mean_mean_part": df["mean_part"].mean(),
        "std_mean_part": df["mean_part"].std()/ np.sqrt(n_rows),
        "mean_inv_std_part": (1. / df["std_part"]).mean(),
        "std_inv_std_part": (1. / df["std_part"]).std()/ np.sqrt(n_rows),
    })

    summary = pd.DataFrame(records)

plt.figure(figsize=(8, 5))
for p, grp in summary.groupby("p"):
    grp = grp.sort_values("m")
    plt.errorbar(
        grp["m"] / n,
        grp["mean_mean_part"],
        yerr=grp["std_mean_part"],
        marker="o",
        label=f"p={p:g}",
    )

plt.xlabel("m/n")
plt.ylabel("Mean of mean_part")
plt.title("Mean Partial Derivative")
plt.legend(title="p")
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(8, 5))
for p, grp in summary.groupby("p"):
    grp = grp.sort_values("m")
    plt.errorbar(
        grp["m"]/n,
        grp["mean_inv_std_part"],
        yerr=grp["std_inv_std_part"],
        marker="o",
        label=f"p={p:g}",
    )

plt.xlabel("m/n")
plt.ylabel("Mean of 1/std_part")
plt.title("Mean Inverse Std Partial Derivative")
plt.legend(title="p")
plt.grid(True)
plt.tight_layout()

plt.show()