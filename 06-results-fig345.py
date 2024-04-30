import mpl_config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df_loss_epoch = pd.read_csv('./export/loss_epoch.csv')
df_test_pred = pd.read_csv('./export/test_pred.csv')
df_loss_epoch = df_loss_epoch.iloc[1:]

epoch_early_stop = np.argmin(df_loss_epoch["loss_val"]) + 1
print(epoch_early_stop)


cm = 1/2.54
# === fig3 ===
fig3, ax3 = plt.subplots(figsize=(10*cm, 10/1.618*cm))
ax3.set_xlabel('Epoch')
ax3.set_ylabel("Mean squared error")
ax3.set_yscale('log')
ax3.set_ylim([400, 1e5])
ax3.plot(df_loss_epoch['loss_train'], label='Train loss')
ax3.plot(df_loss_epoch['loss_val'], label='Validation loss')
ax3.axvline(
    epoch_early_stop,
    linestyle="--",
    color="red",
    label="Minimum validation loss",
)
fig3.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9))
fig3.tight_layout()  # otherwise the right y-label is slightly clipped
fig3.savefig("./figs/fig3.pdf", dpi=300, transparent=True)  # transparent for Adobe Illustrator.


# === fig4 ===
fig4, ax4 = plt.subplots(figsize=(8*cm, 8*cm))
# ax4.scatter(df_test_pred['true_y'], df_test_pred['pred_y'], s=9)
sns.scatterplot(x=df_test_pred['true_y'], y=df_test_pred['pred_y'])
ax4.set_xlabel("Real capacity (mAh)")
ax4.set_ylabel("Predicted capacity (mAh)")
ax4.set_aspect('equal', 'box')
ax4.plot([2400, 3400], [2400, 3400], color='red', label='Real capacity')
ax4.legend()
fig4.tight_layout()  # otherwise the right y-label is slightly clipped
fig4.savefig("./figs/fig4.pdf", dpi=300, transparent=True)  # transparent for Adobe Illustrator.


# === fig5 ===
fig5, ax5 = plt.subplots(figsize=(8*cm, 8*cm))
sns.histplot(df_test_pred['pred_err'], kde=True, bins=30, color='orange')
ax5.set_xlabel("Prediction percentage error (%)")
ax5.set_ylabel("Count")
plt.rc('text', usetex=True)
ax5.text(2.2, 600, r'$-0.03 \pm 0.77$', ha='center', va='center')
fig5.tight_layout()  # otherwise the right y-label is slightly clipped
fig5.savefig("./figs/fig5.pdf", dpi=300, transparent=True)  # transparent for Adobe Illustrator.

