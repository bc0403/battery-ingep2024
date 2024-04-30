import torch
import timeit
import re
import os
from natsort import natsorted, ns
import pandas as pd
import numpy as np


def test_cpu_gpu_speed(size=1000, gpu="mps"):
    """compare the operation speed between cpu and gpu

    Parameters:
    size: size of matrix
    gpu: 'cuda' or 'mps'

    Return: None
    """
    a_cpu = torch.rand(size, device="cpu")
    b_cpu = torch.rand((size, size), device="cpu")
    a_gpu = torch.rand(size, device=gpu)
    b_gpu = torch.rand((size, size), device=gpu)
    print("cpu", timeit.timeit(lambda: a_cpu @ b_cpu, number=100_000))
    print(gpu, timeit.timeit(lambda: a_gpu @ b_gpu, number=100_000))


# test_cpu_gpu_speed(size=2000, gpu="mps")


def extract_battery_info_from_filename(dir, filename):
    """extract information from file name, for battery dataset

    Parameters:
    dir: directory
    filename: file name with mask `CYX-Y_Z-#N.csv`, such as `CY25-1_1-#1.csv`

    Return:
    X: temperature, 'str'
    Y: charge current, 'str'
    Z: discharge current, 'str'
    N: cell id, 'str'
    """
    file = os.path.join(dir, filename)
    # Regular expression pattern to match the file mask
    pattern = r"CY(\d+)-(\d+)_([^#-]+)-#(\d+)\.csv$"
    matched = re.search(pattern, file)
    if matched:
        num1, num2, num3, num4 = matched.groups()
        # print(f"Extracted numbers from {file}: {num1}, {num2}, {num3}, and {num4}")
    else:
        print("No match found")

    return num1, num2, num3, num4


# extract_battery_info_from_filename("Dataset_1_NCA_batterydir", "CY25-05_1-#1.csv")


def test_extract_battery_info_from_filename(dir):
    """test the function of extract_battery_info_from_filename()

    Parameter:
    dir: directory
    """
    files = os.listdir(dir)
    files = natsorted(files, alg=ns.IGNORECASE)  # Sort files naturally

    n = len(files)
    if n > 0:
        print(f"\n=== {n} files found ===")
    else:
        pass

    for file in files:
        extract_battery_info_from_filename(dir, file)


# test_extract_battery_info_from_filename("Dataset_1_NCA_battery")
# test_extract_battery_info_from_filename("Dataset_2_NCM_battery")
# test_extract_battery_info_from_filename("Dataset_3_NCM_NCA_battery")


def extract_features_targets(filename, cond):

    df = pd.read_csv(filename)
    df = df.loc[df["cond"] == cond]

    a = [
        "Capacity",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
    ]
    # Convert multiple columns to float
    df[a] = df[a].astype(float)

    print(df.head(10))

    # Features are in columns one to end
    X = df.iloc[:, 6:20].to_numpy()

    # # Scale features
    # X = StandardScaler().fit_transform(X)

    # Labels are in the column zero
    y = df.iloc[:, 5].to_numpy()

    # return Features and Labels
    return X, y


# X, y = extract_features_targets("Dataset_1_NCA_battery_clean.csv", "CY45-05/1")
# print(X, X.shape)
# print(y, y.shape)
