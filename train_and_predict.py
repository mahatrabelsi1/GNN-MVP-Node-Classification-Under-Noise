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
# 2. PREPARE GRAPH
# =========================

# Features
X = nodes.drop(columns=["node_id"]).values
x = torch.tensor(X, dtype=torch.float)

# Edge index
edge_index = torch.tensor(edges.values.T, dtype=torch.long)

# Labels (initialize with -1)
y = torch.full((len(nodes),), -1, dtype=torch.long)

# Fill train + val labels
y[train_df["node_id"].values] = torch.tensor(train_df["label"].values)
y[val_df["node_id"].values] = torch.tensor(val_df["label"].values)

# Masks
train_mask = torch.zeros(len(nodes), dtype=torch.bool)
val_mask = torch.zeros(len(nodes), dtype=torch.bool)

train_mask[train_df["node_id"].values] = True
val_mask[val_df["node_id"].values] = True

# Graph object
data = Data(x=x, edge_index=edge_index, y=y)

# =========================
# 3. MODEL (Simple GCN)
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
# 4. TRAINING
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
# 5. PREDICTIONS (IMPORTANT)
# =========================
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()  # probability of class 1

# =========================
# 6. CREATE SUBMISSION
# =========================
submission = pd.DataFrame({
    "id": test_df["node_id"],
    "y_pred": probs[test_df["node_id"]]
})

submission.to_csv("predictions.csv", index=False)

print("✅ predictions.csv generated!")
