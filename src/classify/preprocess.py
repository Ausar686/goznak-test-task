from typing import Callable
import os
import os.path as osp

from tqdm import tqdm
import cv2
import numpy as np


def preprocess_wrapper(func: Callable) -> Callable:
    """Prettify console output for preprocessing."""
    def wrapper(*args, **kwargs) -> object:
        dataset_type = kwargs.get("type", None)
        if dataset_type is not None:
            print(f"[INFO]: Preparing {dataset_type} dataset...")
        res = func(*args, **kwargs)
        if dataset_type is not None:
            print(f"[INFO]: Finished preparing {dataset_type} dataset.")
        return res
    return wrapper

def to_image(sound: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """
    Convert & resize a 2D-array into cv2-compatible B/W array.

    'sound' should be a [0.0, 1.0] normalized 2D np.ndarray.
    """
    image_from_sound = (sound.T * 255).astype(np.uint8)
    image = cv2.resize(image_from_sound, (imgsz, imgsz // 16 * 9))
    return image

@preprocess_wrapper
def preprocess(dataset_dir: str, output_dir: str, **kwargs) -> None:
    """
    Convert mel-spectrograms into images and save them to disk.
    
    The following hierarchy is assumed:
        - clean\
        - - speaker1_id\
        - - - spec_sample_1_1.npy
        - - - spec_sample_1_2.npy
        - - - ...
        - - speaker2_id\
        - - - spec_sample_2_1.npy
        - - - spec_sample_2_2.npy
        - - - ...
        - - ...
        - noisy\
        - - speaker1_id\
        - - - spec_sample_1_1.npy
        - - - spec_sample_1_2.npy
        - - - ...
        - - speaker2_id\
        - - - spec_sample_2_1.npy
        - - - spec_sample_2_2.npy
        - - - ...
        - - ...
    """
    for sound_type in os.listdir(dataset_dir):
        sound_dir = osp.join(dataset_dir, sound_type)
        sound_type_out_dir = osp.join(output_dir, sound_type)
        if not osp.isdir(sound_type_out_dir):
            os.makedirs(sound_type_out_dir)
        for speaker_id in tqdm(os.listdir(sound_dir)):
            speaker_dir = osp.join(sound_dir, speaker_id)
            for filename in os.listdir(speaker_dir):
                out_name = f"{osp.splitext(filename)[0]}.png"
                output_path = osp.join(sound_type_out_dir, out_name)
                npy_path = osp.join(speaker_dir, filename)
                sound = np.load(npy_path)
                image = to_image(sound)
                cv2.imwrite(output_path, image)
    return


if __name__ == "__main__":
    GOZNAK_REPO_PATH = os.environ["GOZNAK_REPO_PATH"]

    train_dir = osp.join(GOZNAK_REPO_PATH, "data", "base", "train")
    val_dir = osp.join(GOZNAK_REPO_PATH, "data", "base", "val")
    out_base_dir = osp.join(GOZNAK_REPO_PATH, "data", "mono")

    train_out_dir = osp.join(out_base_dir, "train")
    val_out_dir = osp.join(out_base_dir, "val")

    preprocess(train_dir, train_out_dir, type="train")
    preprocess(val_dir, val_out_dir, type="val")