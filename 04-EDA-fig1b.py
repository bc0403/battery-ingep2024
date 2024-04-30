import pandas as pd

import mpl_config
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset_1_NCA_battery_clean.csv")
print(df.shape)

cm = 1/2.54
fig, ax = plt.subplots(figsize=(10/1.618*cm, 10/1.618*cm))
sns.histplot(df["Capacity"], kde=True, bins=15)
ax.set_xlabel("Capacity (mAh)")
ax.set_xticks([2600, 3000, 3400])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("./figs/fig1b.pdf", dpi=300,  transparent=True)
plt.show()
