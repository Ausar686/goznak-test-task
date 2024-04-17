import os
import os.path as osp

from ultralytics import YOLO

if __name__ == "__main__":
	GOZNAK_REPO_PATH = os.environ["GOZNAK_REPO_PATH"]

	weights = osp.join(GOZNAK_REPO_PATH, "models", "base", "yolov8n-cls.pt")
	data_mono = osp.join(GOZNAK_REPO_PATH, "data", "mono")
	project = osp.join(GOZNAK_REPO_PATH, "models", "classify")
	epochs = 100
	imgsz = 224

	# Adjust these based on your hardware.
	# These parameters were used with NVIDIA RTX 3070 (8 GB VRAM) + 32 GB RAM.
	batch = 192
	workers = 16

	model = YOLO(weights)
	model.train(
		data=data_mono,
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		workers=workers,
		project=project,
	)