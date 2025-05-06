import torch
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(net):
    torch.save(net.state_dict(), './models/trained_model.pth')
    print("Checkpoint saved!")


def load_checkpoint(net, checkpoint_path):
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print("Checkpoint Loaded!")
    return net


def show_images(original, reconstructed):
    # Ensure tensors are on CPU and detached
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu()

    # Flattened? Reshape to (batch, 28, 28)
    if original.ndim == 2 and original.shape[1] == 784:
        original = original.view(-1, 28, 28)
    if reconstructed.ndim == 2 and reconstructed.shape[1] == 784:
        reconstructed = reconstructed.view(-1, 28, 28)

    # Has channel dim? Remove it: (B, 1, 28, 28) -> (B, 28, 28)
    if original.ndim == 4 and original.shape[1] == 1:
        original = original.squeeze(1)
    if reconstructed.ndim == 4 and reconstructed.shape[1] == 1:
        reconstructed = reconstructed.squeeze(1)

    # Pick the first image
    original_img = original[0].numpy()
    reconstructed_img = reconstructed[0].numpy()

    # Plot
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()