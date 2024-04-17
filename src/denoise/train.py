from copy import deepcopy
from typing import Tuple
import os
import os.path as osp
import random
import uuid

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


def set_seed(seed: int) -> None:
    """
    Set seeed for reproducibility.

    To ensure robustness one can use additional approaches, like:
    ```python
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ```
    As well as defining custom 'worker_init_fn' for DataLoader.

    However, in this code we will neglect these measures.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

def get_padding_shape(base_dir: str) -> Tuple[int, int]:
    """Get maximum matrix shape among all .npy files in subfolders."""
    h_max = 0
    w_max = 0
    for stype in ["clean", "noisy"]:
        print("[INFO]: Processsing", stype)
        for speaker in tqdm(os.listdir(f"{base_dir}/{stype}")):
            for filename in os.listdir(f"{base_dir}/{stype}/{speaker}"):
                if not filename.endswith(".npy"):
                    continue
                npy_path = osp.join(base_dir, stype, speaker, filename)
                sound = np.load(npy_path)
                h, w = sound.shape
                w_max, h_max = max(w, w_max), max(h, h_max)
    return h_max, w_max

def pad_to_shape(input_tensor, target_height, target_width):
    # Assuming input_tensor is of shape (1, H, W)
    # Calculate current dimensions
    _, H, W = input_tensor.shape

    # Calculate required padding
    pad_height = (target_height - H) if target_height > H else 0
    pad_width = (target_width - W) if target_width > W else 0

    # Ensure padding is divided equally on both sides, but bias any odd padding to 'bottom' or 'right'
    padding = (pad_width // 2, pad_width - pad_width // 2,  # Left, Right
               pad_height // 2, pad_height - pad_height // 2)  # Top, Bottom

    # Pad the tensor and maintain a single channel dimension
    padded_tensor = F.pad(input_tensor, padding, "constant", 0)

    return padded_tensor

class Autoencoder(nn.Module):
    """
    Autoencoder class for denoising.

    Usage:
        model = Autoencoder()
        noisy_sound = torch.from_numpy(np.load("/path/to/noisy"))
        denoised_sound = model(noisy_sound)
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3),
            nn.Sigmoid()  # Using Sigmoid to scale the output between 0 and 1
        )
        return

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MelSpectrogramDataset(Dataset):
    """Mel-spectrogram dataset implementation as Dataset subclass."""
    def __init__(self, clean_dir, noisy_dir, transform=None):
        """
        Args:
            clean_dir (string): Directory with all the clean spectrograms.
            noisy_dir (string): Directory with all the noisy spectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        speakers = os.listdir(clean_dir)
        speakers_noisy = os.listdir(noisy_dir)
        assert speakers == speakers_noisy
        self.clean_files = [osp.join(clean_dir, speaker, f) 
                            for speaker in speakers for f in os.listdir(osp.join(clean_dir, speaker))]
        self.noisy_files = [osp.join(noisy_dir, speaker, f) 
                            for speaker in speakers for f in os.listdir(osp.join(noisy_dir, speaker))]
        self.transform = transform
        assert len(self.clean_files) == len(self.noisy_files)
        return

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]
        
        clean = np.load(clean_path)
        noisy = np.load(noisy_path)
        
        if self.transform is not None:
            clean = self.transform(clean)
            noisy = self.transform(noisy)
        
        return (clean, noisy)

class SpectrogramTransform(object):
    """
    Callable class for spectrogram transform.

    Init:
        transform = SpectrogramTransform(height, width, dtype=torch.float16)
    Usage (numpy):
        matrix = np.random.random((h, w)) # or (c, h, w)
        transformed = transform(matrix) # shape: (c, self.height, self.width)
    Usage (torch):
        tensor = torch.randn((h, w)) # or (c, h, w)
        transformed = transform(tensor) # shape (c, self.height, self.width)
    Returns: 
        transformed (torch.Tensor)
    """
    def __init__(self, height: int, width: int, **kwargs):
        self.height = height
        self.width = width
        self.dtype = kwargs.get("dtype", torch.float16)
        return
    
    def __call__(self, spectrogram: np.ndarray) -> torch.Tensor:
        if isinstance(spectrogram, np.ndarray):
            spectorgram = torch.from_numpy(spectrogram)
        if len(spectrogram.shape) == 2: # (h, w) - > (c, h. w)
            spectrogram = spectorgram.unsqueeze(0)
        spectrogram = pad_to_shape(spectrogram, self.height, self.width).to(self.dtype)
        return spectrogram

def train(
    base_dir: str,
    *,
    out_dir: str = "models",
    epochs: int = 50,
    device: int|str|torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    batch: int = 32,
    workers: int = 4,
    save_period: int = -1,
    patience: int = 50,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    **kwargs
) -> Autoencoder:
    # Set random seed
    set_seed(seed)

    # Create the model instance
    model = Autoencoder()
    model.to(device).to(dtype).train()
    print(model)  # Print the structure of the network

    # Get padding shapes
    print("Obtaining padding shape...")
    h_max, w_max = get_padding_shape(base_dir)
    print("Padding shape:", (h_max, w_max))

    # Set up the optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Set up save path for checkpoints
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    suffix = str(uuid.uuid4())
    save_path = osp.join(out_dir, f"model_{suffix}.pt")

    # Initialize dataloader and dataset
    clean_dir = osp.join(base_dir, "clean")
    noisy_dir = osp.join(base_dir, "noisy")
    transform = SpectrogramTransform(h_max, w_max, dtype=dtype)
    dataset = MelSpectrogramDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print("Dataset contains", len(dataset), "pairs of images.")

    # Training loop
    min_loss = float("inf")
    best_model = model
    best_epoch = -1
    print("Training progress:")
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        train_loss = 0.0
        passed = 0
        for clean, noisy in tqdm(dataloader):
            optimizer.zero_grad()

            # Forward
            clean, noisy = clean.to(device), noisy.to(device)
            output = model(noisy)
            loss = criterion(output, clean)

            # Backward
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Visualize current state
            passed += clean.shape[0]
            pbar.set_postfix(epoch=epoch+1, loss=f"{train_loss/passed:.5f}")

        # Update best model
        if train_loss < min_loss:
            min_loss = train_loss
            best_epoch = epoch
            # As we have no layers with buffers (like BN layer)
            # we can simply use deepcopy to store our best model
            best_model = deepcopy(model)
            torch.save(best_model, save_path)

        # Early stopping
        if epoch - best_epoch > patience:
            print(f"No improvement for {best_epoch-epoch} epochs. Stopping...")
            break

        # Save intermediate checkpoint if necessary 
        if save_period > 0 and (epoch + 1) % save_period == 0:
            ckpt_path = osp.join(out_dir, f"ckpt_{suffix}_{epoch+1}.pt")
            torch.save(model, ckpt_path)
    pbar.close()

    # Save the best model
    best_model.eval()
    torch.save(best_model, save_path)
    print("[INFO]: Best loss:", min_loss)
    return best_model


if __name__ == "__main__":
    GOZNAK_REPO_PATH = os.environ["GOZNAK_REPO_PATH"]

    # Set training params
    base_dir = osp.join(GOZNAK_REPO_PATH, "data", "base", "train")
    out_dir = osp.join(GOZNAK_REPO_PATH, "models", "denoise")
    device = 0
    dtype = torch.float16
    epochs = 20
    save_period = 5

    train(
        base_dir,
        out_dir=out_dir,
        device=device,
        dtype=dtype,
        epochs=epochs,
        save_period=save_period,
    )