import torch
import torch.nn as nn

class InefficientModel(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.clone() * 1.0
        return self.layers(x).squeeze(-1)


def load_model(path: str = "inefficient_model.pt", device: str = "cpu") -> nn.Module:
    """
    Load the model from the given path.
    Only accepts pickles in the form of state dicts, to avoid pickling errors.
    """
    model = InefficientModel(in_dim=3)
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = InefficientModel(3)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
    torch.save(model.state_dict(), "inefficient_model.pt")
    print("Saved inefficient_model.pt")