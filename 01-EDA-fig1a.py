import numpy as np
import pandas as pd
import os
import mpl_config
import matplotlib.pyplot as plt

DATA_PATH = "./Dataset_1_NCA_battery/"  # raw data folder
FILE_NAME = "CY45-05_1-#1.csv"

df = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))
df1 = df.loc[df["cycle number"] == 1]
df2 = df.loc[
    (df["cycle number"] == 1) & (df["control/V/mA"] == 0)
]  # for relaxation voltage

# there are two relaxation periods (relaxation after charing & relaxation after discharging)
# pick out the first relaxation period
# Find the first non-continuous point in the index
index_diff = np.diff(df2.index)  # Calculate the difference between consecutive indices
discontinuity_index = np.where(index_diff > 1)[0]  # Find where diff > 1

if len(discontinuity_index) > 0:
    # If there is a discontinuity, select up to the first discontinuous index
    first_discontinuous_index = discontinuity_index[0]
    continuous_sub_df = df2.iloc[
        : first_discontinuous_index + 1
    ]  # include the first_discontinuous_index
else:
    # If the indices are all continuous, select the entire DataFrame
    continuous_sub_df = df2

df3 = continuous_sub_df  # relaxation after charging

duration = df3["time/s"].max() - df3["time/s"].min()
points = df3.shape[0]
print(f"exctracted relaxation period: {duration:.1f} s; {points} points")

# index = np.where(np.abs(df3['control/V/mA']) == 0)
# print(index)
cm = 1/2.54
fig, ax1 = plt.subplots(figsize=(10*cm, 10/1.618*cm))

color1 = "tab:blue"
ax1.plot(df1["time/s"], df1["Ecell/V"], color=color1, label="Voltage")
ax1.tick_params(axis="y", labelcolor=color1)
# ax1.set_ylim(2.5, 4.5)
ax1.set_ylabel("Voltage (V)", color=color1)
ax1.set_yticks([2.6, 3.0, 3.4, 3.8, 4.2])
ax1.scatter(df3["time/s"], df3["Ecell/V"], color="tab:green", label="Relaxation")
ax1.set_xlabel("Time (s)")

color2 = "tab:red"
ax2 = ax1.twinx()
ax2.plot(df1["time/s"], df1["<I>/mA"], color=color2, label="Current", linestyle="--")
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylabel("Current (mA)", color=color2)

# Create a single legend with entries from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="lower left")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("./figs/fig1a.pdf", dpi=300, transparent=True)  # transparent for Adobe Illustrator.
# plt.show()
