"""1D CNN over a window of daily features, with two heads: regime class and log volatility."""
from torch import nn

class CNN(nn.Module):
    def __init__(self, n_feat=7, window=60, ch=32, kernel=5, hidden=32, dropout=0.2, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, ch, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, ch, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                nn.Linear(ch * (window // 4), hidden), nn.ReLU())   # two pools -> window // 4 steps
        self.regime_head = nn.Linear(hidden, n_classes)
        self.vol_head = nn.Linear(hidden, 1)

    def forward(self, x):                                  # x: (batch, window, n_feat)
        h = self.fc(self.conv(x.transpose(1, 2)))          # Conv1d wants (batch, channels, time)
        return self.regime_head(h), self.vol_head(h).squeeze(-1)

def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
