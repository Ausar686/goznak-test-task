from os import listdir as ls
import os
import os.path as osp

from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

from preprocess import to_image


def eval(model: YOLO|str, spectrogram: np.ndarray|str) -> str|float:
    """
    Evaluate sound classification model.

    Args:
        model: (YOLO|str) - either YOLO model or path to it's weights (str).
        spectrogram: (np.ndarray|str) - should be either of the following:
            1. 2D np.ndarray: then evaluate on a given array 
                and return sound type ('clean' or 'noisy') as string.
            2. Path to .npy file (str). The load .npy file and execute as 1.
            3. Path to test directory (str). Then calculate accuracy 
                on test dataset and return it as float value.
                If dataset is empty, return 0.0.
    Returns:
        sound_type (str) or accuracy (float).
    """

    def _eval(model, spectrogram: np.ndarray|str) -> str:
        if isinstance(spectrogram, str):
            spectrogram = np.load(spectrogram)
        image = to_image(spectrogram)
        res = model(image, verbose=False)[0]
        sound_type = res.names[res.probs.top1]
        return sound_type

    # Load YOLO model if path to weights is given
    if isinstance(model, str):
        model = YOLO(model)

    # Process test directory
    if isinstance(spectrogram, str) and osp.isdir(spectrogram):
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
        print("[INFO]: Started evaluation on:", spectrogram)
        ok = 0
        total = 0
        for stype in ls(spectrogram):
            print("[INFO]: Processing", stype)
            pbar = tqdm(ls(osp.join(spectrogram, stype)))
            for speaker in pbar:
                for filename in ls(osp.join(spectrogram, stype, speaker)):
                    path = osp.join(spectrogram, stype, speaker, filename)
                    stype_pred = _eval(model, path)
                    ok += stype_pred == stype
                    total += 1
                    pbar.set_postfix(accuracy=f"{ok/total*100:.2f}%")
        return ok / total if total > 0 else 0.0

    # Otherwise process a single spectrogram in demo mode
    # and return sound type ('clean' or 'noisy') as string.
    else:
        sound_type = _eval(model, spectrogram)
        return sound_type 


if __name__ == "__main__":
    GOZNAK_REPO_PATH = os.environ["GOZNAK_REPO_PATH"]

    weights = osp.join(GOZNAK_REPO_PATH, "models", "classify", "train", "weights", "best.pt")
    test_dir = osp.join(GOZNAK_REPO_PATH, "data", "base", "val") # Your path to test dir here

    model = YOLO(weights)
    acc = eval(model, test_dir)
    print("[INFO]: Test accuracy:", acc)