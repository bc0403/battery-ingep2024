import pandas as pd
import os
import re
import numpy as np
from natsort import natsorted, ns  # for natually sort

from myutils import extract_battery_info_from_filename  # extract info from file name


DATA_PATH = "./Dataset_1_NCA_battery/"

df_list = []  # List to collect DataFrames
df_res = pd.DataFrame(
    columns=["cond", "cell_id", "cycle", "Voltages", "rate", "Tem", "Capacity"]
)

files = os.listdir(DATA_PATH)
files = natsorted(files, alg=ns.IGNORECASE)  # Sort files naturally

n = len(files)
# n = 3
for file in range(n):
    print(f"processing file {file+1}/{n} ...")
    # Tem = int(files[file][2:4])
    Tem, ccr, dcr, cell_id = extract_battery_info_from_filename(DATA_PATH, files[file])
    data_r = pd.read_csv(os.path.join(DATA_PATH, files[file]))
    for i in range(
        int(np.min(data_r["cycle number"].values)),
        int(np.max(data_r["cycle number"].values)) + 1,
    ):
        data_i = data_r[data_r["cycle number"] == i]
        Ecell = np.array(data_i["Ecell/V"])
        Q_dis = np.array(data_i["Q discharge/mA.h"])
        Current = np.array(data_i["<I>/mA"])
        control = np.array(data_i["control/V/mA"])
        cr = np.array(data_i["control/mA"])[1] / 3500
        if np.max(Q_dis) < 2500 or np.max(Q_dis) > 3500:
            continue
        index = np.where(np.abs(control) == 0)
        start = index[0][0]
        end = 13
        for j in range(3):
            if control[start + 3] == 0:
                break
            else:
                start = index[0][j + 1]
        if Current[start] > 1:
            start = start + 1
            if control[start + 13] != 0:
                end = 12
        if control[start + end] == 0 and Ecell[start + end] > 4.0:
            df_list.append(
                pd.DataFrame(
                    {
                        "cond": ["CY" + Tem + "-" + ccr + "/" + dcr],
                        "cell_id": [cell_id],
                        "cycle": [i],
                        "Voltages": [list(Ecell[start : start + 14])],
                        "rate": [cr],
                        "Tem": [Tem],
                        "Capacity": [np.max(Q_dis)],
                    }
                )
            )

print(f"concat data ...")
# Concatenate all DataFrames in the list
df_res = pd.concat(df_list, ignore_index=True)

# # Save to excel file
# df_res.to_excel("Dataset_1_NCA_battery.xlsx", index=False)

# save to csv file
df_res.to_csv("Dataset_1_NCA_battery.csv", index=False)
print("=== Features extraction is done, and here is the summary: ===")
print(f"{len(df_res)} samples extracted")
agg_func = {"cell_id": ["nunique"], "cycle": ["count"]}
print(df_res.groupby(["cond"]).agg(agg_func))
