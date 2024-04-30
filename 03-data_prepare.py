import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import json  # To convert string representation of list to actual list

df = pd.read_csv('Dataset_1_NCA_battery.csv')

# Convert string representation of list to actual list using json.loads
df['Voltages'] = df['Voltages'].apply(json.loads)
print(df.head(30))

# Create a new DataFrame from the lists in 'Column1'
expanded_df = pd.DataFrame(df['Voltages'].tolist())

# Concatenate the new DataFrame with the original DataFrame
df = pd.concat([df, expanded_df], axis=1)

# Drop the original column with lists (optional)
df = df.drop('Voltages', axis=1)

print(df.describe())
df.to_csv('Dataset_1_NCA_battery_clean.csv', index=False)

 