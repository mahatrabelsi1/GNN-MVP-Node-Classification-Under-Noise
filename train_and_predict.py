import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# =========================
# 1. LOAD DATA
# =========================
nodes = pd.read_csv("data/public/nodes.csv")
edges = pd.read_csv("data/public/edges.csv")
train_df = pd.read_csv("data/public/train.csv")
val_df = pd.read_csv("data/public/val.csv")
test_df = pd.read_csv("data/public/test_nodes.csv")

# =========================
# 2. HANDLE NODE IDs SAFELY
# =========================

# Detect node id column
if "node_id" in nodes.columns:
    node_id_col = "node_id"
elif "id" in nodes.columns:
    node_id_col = "id"
else:
    node_id_col = None

# Features
if node_id_col:
    X = nodes.drop(columns=[node_id_col]).values
else:
    X = nodes.values

x = torch.tensor(X, dtype=torch.float)

# =========================
# 3. EDGE INDEX
# =========================
edge_index = torch.tensor(edges.values.T, dtype=torch.long)

# =========================
# 4. HANDLE TRAIN/VAL IDS
# =========================

def get_id_col(df):
    if "node_id" in df.columns:
        return "node_id"
    elif "id" in df.columns:
        return "id"
    else:
        raise ValueError("No ID column found!")

train_id_col = get_id_col(train_df)
val_id_col = get_id_col(val_df)
test_id_col = get_id_col(test_df)

# =========================
# 5. LABEL VECTOR
# =========================
num_nodes = len(nodes)
y = torch.full((num_nodes,), -1, dtype=torch.long)

y[train_df[train_id_col].values] = torch.tensor(train_df["label"].values)
y[val_df[val_id_col].values] = torch.tensor(val_df["label"].values)

# Masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_df[train_id_col].values] = True
val_mask[val_df[val_id_col].values] = True

data = Data(x=x, edge_index=edge_index, y=y)

# =========================
# 6. MODEL
# =========================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(x.shape[1], 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 7. TRAINING
# =========================
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = out[val_mask].argmax(dim=1)
        val_acc = (val_pred == data.y[val_mask]).float().mean()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

# =========================
# 8. PREDICTIONS
# =========================
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

# =========================
# 9. SUBMISSION FILE
# =========================
submission = pd.DataFrame({
    "id": test_df[test_id_col],
    "y_pred": probs[test_df[test_id_col]]
})

submission.to_csv("predictions.csv", index=False)

print("✅ predictions.csv generated successfully!")
