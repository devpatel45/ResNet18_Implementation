from resblock import ResidualBlock
import torch
if __name__ == "__main__":
    block = ResidualBlock(64, 64)
    x = torch.randn(1, 64, 32, 32)  # batch=1, channels=64, image=32x32
    out = block(x)
    print(out.shape)  # Should be [1, 64, 32, 32]
