from os import listdir as ls
import os
import os.path as osp

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from train import Autoencoder


def eval(
    model: Autoencoder,
    noisy: np.ndarray|str,
    *,
    save_path: str = None,
    output_type: str = "numpy"
) -> np.ndarray|torch.Tensor|float:
    """
    Evaluate sound denoising model.

    Args:
        model: (Autoencoder) - sound denoising model.
        noisy: (np.ndarray|str) - should be either of the following:
            1. 2D np.ndarray: then evaluate on a given array 
                return denoised sound and save it if necessary.
            2. Path to .npy file (str). The load .npy file and execute as 1.
            3. Path to test directory (str). Then calculate MSE 
                on test dataset and return it as float value.
                If dataset is empty, return 0.0.
        save_path: (str) - path to save denoised numpy sound (or None).
            NOTE: Setting 'output_type'!="numpy" force ignores 'save_path'.
            NOTE: Does not affect if 'noisy' is a test directory path.
            Default: None.
        output_type:(str) - type of output (either torch.Tensor or np.ndarray).
            NOTE: Does not affect if 'noisy' is a test directory path.
            Default: "numpy".
    Returns:
        One of the following:
            1. 'denoised' (np.ndarray|torch.Tensor) - if 'noisy' is not a test directory path.
            2. 'mse' (float) - if 'noisy' is a test directory path.
    """

    def _transform(spectrogram: np.ndarray|str, model: Autoencoder) -> torch.Tensor:
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.from_numpy(spectrogram)
        elif isinstance(spectrogram, str):
            spectrogram = torch.from_numpy(np.load(spectrogram))
        param = next(model.named_parameters())[1]
        if spectrogram.device != param.device:
            spectrogram = spectrogram.to(param.device)
        if spectrogram.dtype != param.dtype:
            spectrogram = spectrogram.to(param.dtype)
        if len(spectrogram.shape) == 2: # (h, w) -> (c, h, w)
            spectrogram = spectrogram.unsqueeze(0)
        return spectrogram

    def _eval(model: Autoencoder, noisy: np.ndarray|str, *, output_type: str) -> np.ndarray|torch.Tensor:
        # if isnstance(noisy, np.ndarray):
        #     noisy = torch.from_numpy(noisy)
        # elif isinstance(noisy, str):
        #     noisy = torch.from_numpy(np.load(noisy))
        # param = next(model.named_parameters())[1]
        # if noisy.device != param.device:
        #     noisy.to(param.device)
        # if noisy.dtype != param.dtype:
        #     noisy.to(param.dtype)
        # if len(noisy.shape) == 2: # (h, w) -> (c, h, w)
        #     noisy = noisy.unsqueeze(0)
        noisy = _transform(noisy, model)
        output = model(noisy)
        if output_type == "numpy":
            output = output.squeeze(0).cpu().numpy()
        return output

    # Process test directory
    if isinstance(noisy, str) and osp.isdir(noisy):
        # The following hierarchy is assumed:
        # base_directory
        # - clean\
        # - - speaker1_id\
        # - - - spec_sample_1_1.npy
        # - - - spec_sample_1_2.npy
        # - - - ...
        # - - speaker2_id\
        # - - - spec_sample_2_1.npy
        # - - - spec_sample_2_2.npy
        # - - - ...
        # - - ...
        # - noisy\
        # - - speaker1_id\
        # - - - spec_sample_1_1.npy
        # - - - spec_sample_1_2.npy
        # - - - ...
        # - - speaker2_id\
        # - - - spec_sample_2_1.npy
        # - - - spec_sample_2_2.npy
        # - - - ...
        # - - ...
        print("[INFO]: Started evaluation on:", noisy)
        loss = 0
        total = 0
        criterion = nn.MSELoss()
        pbar = tqdm(ls(osp.join(noisy, "noisy")))
        for speaker in pbar:
            for filename in ls(osp.join(noisy, "noisy", speaker)):
                noisy_path = osp.join(noisy, "noisy", speaker, filename)
                clean_path = osp.join(noisy, "clean", speaker, filename)
                denoised = _eval(model, noisy_path, output_type="torch")
                clean = _transform(clean_path, model)
                loss += criterion(denoised, clean)
                total += 1
                pbar.set_postfix(mse=f"{loss/total:.5f}")
        return loss / total if total > 0 else 0.0

    # Otherwise process a single spectrogram in demo mode
    # and save results if necessary.
    else:
        denoised = _eval(model, noisy, output_type=output_type)
        if save_path is not None and output_type == "numpy":
            np.save(denoised, save_path)
        return denoised


if __name__ == "__main__":
    GOZNAK_REPO_PATH = os.environ["GOZNAK_REPO_PATH"]

    # Set evaluation params
    weights = osp.join(GOZNAK_REPO_PATH, "models", "denoise", "model.pt")
    test_dir = osp.join(GOZNAK_REPO_PATH, "data", "base", "val")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(weights).to(device)
    mse = eval(model, test_dir)
    print("[INFO]: MSELoss on test set:", mse)