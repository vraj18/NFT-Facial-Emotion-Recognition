# Emotion Recognition (PyTorch + MTCNN + FER2013)

Steps:
1. Create venv and install: `pip install -r requirements.txt`
2. Place `fer2013.csv` into `data/fer2013.csv`
3. (Optional) Preprocess & cache faces:
   `python src/prepare_faces.py --csv data/fer2013.csv --out cached_faces --img-size 48`
4. Train:
   `python src/train.py --data-csv data/fer2013.csv --cached-dir cached_faces --epochs 25`
5. Inference:
   `python src/infer.py --weights checkpoints/best.pth --source webcam`

Notes:
- The project uses `facenet-pytorch` for MTCNN face detection (PyTorch implementation).
- If you skip caching, MTCNN will run on-the-fly (slower).
- Tune hyperparams in `train.py` as desired.
