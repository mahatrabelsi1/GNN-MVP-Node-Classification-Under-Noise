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
# 2. FEATURES
# =========================
# nodes.csv has 'id' column → remove it
X = nodes.drop(columns=["id"]).values
x = torch.tensor(X, dtype=torch.float)

# =========================
# 3. GRAPH STRUCTURE
# =========================
edge_index = torch.tensor(edges.values.T, dtype=torch.long)

# =========================
# 4. LABELS
# =========================
num_nodes = len(nodes)

y = torch.full((num_nodes,), -1, dtype=torch.long)

# IMPORTANT: label column is 'diabetes'
y[train_df["id"].values] = torch.tensor(train_df["diabetes"].values)
y[val_df["id"].values] = torch.tensor(val_df["diabetes"].values)

# Masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_df["id"].values] = True
val_mask[val_df["id"].values] = True

data = Data(x=x, edge_index=edge_index, y=y)

# =========================
# 5. MODEL (Simple GCN)
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
# 6. TRAINING
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
# 7. PREDICTIONS (PROBABILITIES)
# =========================
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

# =========================
# 8. CREATE SUBMISSION
# =========================
submission = pd.DataFrame({
    "id": test_df["id"],
    "y_pred": probs[test_df["id"]]
})

submission.to_csv("predictions.csv", index=False)

print("✅ predictions.csv generated successfully!")

# =========================
# 9. SANITY CHECK
# =========================
print("Min prob:", probs.min())
print("Max prob:", probs.max())
print("Total test samples:", len(test_df))
