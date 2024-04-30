import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from myutils import extract_features_targets


print(f"PyTorch version: {torch.__version__}")

# get gpu, mps, or cpu for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available()  # sometime slower than cpu
        else 
        "cpu"
    )
)
print(f"using {device} device")

# random seed
torch.manual_seed(42)

# constants
BATCH_SIZE = 32
N_EPOCHS = 100
LEARNING_RATE = 1e-3

# global variables
best_mse = np.inf  # hold the best model
best_weights = None  # hold the best model
loss_train = [None]  # Initialize with None to fill index 0
loss_val = [None]  # Initialize with None to fill index 0


def plot_history(loss_train, loss_val):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.plot(loss_train, label="Train")
    plt.plot(loss_val, label="Validation")
    plt.axvline(
        np.argmin(loss_val[1:]) + 1,
        linestyle="--",
        color="red",
        label="Min Validation Loss",
    )
    plt.legend()
    plt.yscale("log")
    # plt.ylim([0, max(loss_val[1:])])
    plt.tight_layout()
    plt.savefig("./temp/fig3.pdf", dpi=300)
    plt.show()


def plot_prediction(test_labels, test_predictions):
    plt.figure()
    true_value = test_labels.detach().numpy()
    pred_value = test_predictions.detach().numpy()
    # true_value = test_labels
    # pred_value = test_predictions
    plt.scatter(true_value, pred_value)
    plt.xlabel("Real capacity (mAh)")
    plt.ylabel("Predicted capacity (mAh)")
    plt.axis("equal")
    # plt.xlim(plt.xlim())
    # plt.ylim(plt.ylim())
    _ = plt.plot([2400, 3500], [2400, 3500], color="r", label="real")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./temp/fig4.pdf", dpi=300)
    plt.show()

    plt.figure()
    error = (pred_value - true_value) / true_value * 100
    sns.histplot(error, kde=True, bins=50)
    # plt.hist(error, bins = 50)
    plt.xlabel("Prediction Percentage Error (%)")
    _ = plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("./temp/fig5.pdf", dpi=300)
    plt.show()


# ==============================
# 01, prepare the data
# ==============================
X, y = extract_features_targets("Dataset_1_NCA_battery_clean.csv", "CY45-05/1")
print(f"shape of X: {X.shape} | {X.dtype}")
print(f"shpae of y: {y.shape} | {y.dtype}")

# train-test split
X_train_temp, X_test_raw, y_train_temp, y_test = train_test_split(
    X, y, train_size=0.8, shuffle=True, random_state=42
)

# train-validation split
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, train_size=0.8, shuffle=True, random_state=42
)

print(f"shape of X_train_raw: {X_train_raw.shape}")
print(f"shpae of y_train: {y_train.shape}")
print(f"shape of X_val_raw: {X_val_raw.shape}")
print(f"shpae of y_val: {y_val.shape}")
print(f"shape of X_test_raw: {X_test_raw.shape}")
print(f"shpae of y_test: {y_test.shape}")

# standardization
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print(f"mean: {np.mean(X_train[:,0]):.3f}; var: {np.var(X_train[:,0]):.3f}")
print(f"mean: {np.mean(X_val[:,0]):.3f}; var: {np.var(X_val[:,0]):.3f}")
print(f"mean: {np.mean(X_test[:,0]):.3f}; var: {np.var(X_test[:,0]):.3f}")

# convert to pytorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # add channel dim
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


class BatteryNCA(Dataset):
    """
    A custom Dataset class must implement three functions: __init__, __len__, and __getitem__
    """

    def __init__(self, X, y) -> None:
        super().__init__()
        # remember features and target
        self.X = X
        self.y = y

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target


# create Dateset
train_dataset = BatteryNCA(X_train, y_train)
val_dataset = BatteryNCA(X_val, y_val)
test_dataset = BatteryNCA(X_test, y_test)

# create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)  # test one by one
print(f"len of train dataloader: {len(train_dataloader)}")
print(f"len of train dataloader.dataset: {len(train_dataloader.dataset)}")
for X_batch, y_batch in train_dataloader:
    print(X_batch, y_batch)
    break


# ==============================
# 02, build models
# ==============================
# for regressioin
# 1. no activation at the output layer
# 2. use mean square error for loss func
class CompactCNN1D(nn.Module):
    """
    A compact CNN neural network.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print(f"Output shape after conv1: {x.shape}")
        x = self.pool(self.relu(self.conv2(x)))
        # print(f"Output shape after conv1: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"Output shape after conv1: {x.shape}")
        x = self.relu(self.fc1(x))
        # print(f"Output shape after conv1: {x.shape}")
        x = self.fc2(x)
        # print(f"Output shape after conv1: {x.shape}")
        return x


model = CompactCNN1D().to(device)
print(model)
for name, param in model.named_parameters():
    print(f"{name} has {param.numel()} parameters")  # numel() gives the total number of elements in the tensor



# ==============================
# 03, loss function & optimizer
# ==============================
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ==============================
# 03, train the model
# ==============================
def train(dataloader, model, loss_fn, optimizer):
    global loss_train
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()  # sets the model to training mode
    loss_epoch = 0  # train loss of each epoch, averaged by number of batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward pass
        optimizer.zero_grad()  # Reset gradients to zero
        loss.backward()  # Compute gradient of the loss with respect to model parameters

        # update weights
        optimizer.step()  # Update parameters based on the current gradient

        loss_epoch += loss.item()
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    loss_epoch /= num_batches  # average by batches
    loss_train.append(loss_epoch)
    print(f"training loss (mse): {loss_epoch:>8f}")


def validation(dataloader, model, loss_fn):
    global loss_val, best_mse, best_weights
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # sets the model to testing mode
    loss_epoch = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_epoch += loss_fn(pred, y).item()
    loss_epoch /= num_batches
    loss_val.append(loss_epoch)
    print(f"validation loss (mse): {loss_epoch:>8f}\n")
    if loss_epoch < best_mse:
        best_mse = loss_epoch
        best_weights = copy.deepcopy(model.state_dict())


# train and validation
for epoch in range(1, N_EPOCHS + 1):
    print(f"====== Epoch {epoch} ======")
    train(train_dataloader, model, loss_fn, optimizer)
    validation(val_dataloader, model, loss_fn)
print("Done!")

# test
model.load_state_dict(best_weights)
torch.save(model.state_dict(), "model_weights.pth")  # save the model
model.eval()  # sets the model to testing mode
pred_y = []
pred_err = []
true_y = []
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        pred_y.append(pred.cpu())
        err = (pred - y) / y * 100  # percentage error
        pred_err.append(err.cpu())
        true_y.append(y.cpu())


# save dataframe for figure plot
df_loss_epoch = pd.DataFrame({
    "loss_train": loss_train,
    "loss_val": loss_val
    })
df_test_pred = pd.DataFrame({
    "true_y": [t.item() for t in true_y],
    "pred_y": [t.item() for t in pred_y],
    "pred_err": [t.item() for t in pred_err],
})
df_loss_epoch.to_csv('./export/loss_epoch.csv', index=False)
df_test_pred.to_csv('./export/test_pred.csv', index=False)


print(f"best MSE in validation: {best_mse:.2f}")
print(f"best RMSE in validation: {np.sqrt(best_mse):.2f}")
print(f"early stop epoch: {np.argmin(loss_val[1:])+1}")
print(
    f"prediction percentage error (%): mean {np.mean(pred_err):.3f}; std {np.std(pred_err):.3f}"
)

plot_prediction(torch.Tensor(true_y), torch.Tensor(pred_y))
plot_history(loss_train, loss_val)
