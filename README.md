# Deep Learning Project — DEiT chest X-ray classifier (Iteration 1)

One-sentence: Training and evaluation pipeline that fine-tunes a pretrained DEiT base model on a chest X-ray two-class dataset (NORMAL vs PNEUMONIA), with richer visualizations and robust evaluation (AUC/AP, ROC/PR, confusion heatmaps, sample predictions).

Contents (what this repository provides)

Training / evaluation script (the notebook/script you ran).

Stratified train/val split from the original train/ folder (10% val).

Weighted cross-entropy to handle class imbalance.

Early stopping on validation AUC and multiplicative LR scheduling.

Saved best model checkpoint: /kaggle/working/best_deit_model_fixed.pth

Visualizations saved to /kaggle/working/figs/:

test_roc_pr.png — ROC + PR curves (with optimal ROC point)

training_curves.png — loss / acc / val AUC across epochs

test_confusion_matrix.png — confusion matrix heatmap

class_distribution.png — bar chart for class counts (train/val/test)

sample_predictions_test.png — sample test images with predictions

Final test metrics JSON: /kaggle/working/test_metrics_fixed_full.json

# Quick project summary

Model: deit_base_patch16_224 from timm with a replaced 2-class head (+ dropout).

Loss: CrossEntropyLoss with class weights computed from training split (inverse frequency).

Optimizer: Adam (small LR by default 1e-5) with weight_decay and multiplicative LR scheduler (MUL_LR_FACTOR).

Early stopping: monitor validation AUC (patience default 6 epochs).

Transforms: standard ImageNet normalization; stronger train augmentations: RandomResizedCrop, RandomHorizontalFlip, RandomRotation.

Evaluation: AUC (ROC), AP (precision-recall), accuracy, precision, recall, specificity, confusion matrix.

# Requirements

Minimum Python packages used in the script:

python (3.8+ recommended)
torch
torchvision
timm
scikit-learn
numpy
matplotlib
tqdm


Example install:

pip install torch torchvision timm scikit-learn numpy matplotlib tqdm


Notes: The runtime log shows Torch: 2.6.0+cu124 and timm: 1.0.19. If you run on Kaggle the CUDA + torch build might differ — check compatibility with your GPU.

Dataset / expected folder structure

Place your chest X-ray data root at DATA_ROOT (default in the script: /kaggle/input/xraydata/chest_xray) and structure like:

/kaggle/input/xraydata/chest_xray/
  ├─ train/
  │   ├─ NORMAL/
  │   └─ PNEUMONIA/
  ├─ val/      # optional, this script ignores uploaded val and builds its own stratified val from `train/`
  └─ test/
      ├─ NORMAL/
      └─ PNEUMONIA/


The script uses ImageFolder and a stratified split from train/ (10% -> val) to preserve class balance.

How to run

Assuming your script filename is train_deit.py (or Jupyter notebook), from a shell:

# (optionally) create venv and install requirements
python train_deit.py


If running on Kaggle: ensure dataset is available at /kaggle/input/xraydata/chest_xray and run the notebook / script there.

Key configuration variables (top of script)

You can edit them directly:

SEED = 42
QUICK_RUN = False          # True runs fewer epochs / smaller batch for debugging
BATCH_SIZE = 16
LR = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 20
IMG_SIZE = 224
MUL_LR_FACTOR = 0.995
PATIENCE = 6
VAL_SPLIT = 0.10
DATA_ROOT = '/kaggle/input/xraydata/chest_xray'
SAVE_PATH = '/kaggle/working/best_deit_model_fixed.pth'
FIG_DIR = '/kaggle/working/figs'


Change these to tune experiment length, batch size, learning rate, etc.

Files produced by the run

/kaggle/working/best_deit_model_fixed.pth — best checkpoint (epoch, model_state_dict, optimizer_state_dict)

/kaggle/working/test_metrics_fixed_full.json — saved JSON with test metrics (argmax & prob>=0.5)

/kaggle/working/figs/* — visualizations described above

Example test results (from most recent run)

(these values are from the run log you provided; include them here for reproducibility)

Test metrics (argmax logits):
{
  "acc": 0.95032,
  "prec": 0.94321,
  "recall": 0.97949,
  "f1": 0.96101,
  "auc": 0.98465,
  "ap": 0.98858,
  "specificity": 0.90171,
  "confusion": [211, 23, 8, 382]
}


Interpretation: high AUC/AP, strong sensitivity (recall ≈ 97.95%), slightly reduced specificity (~90.17%).

Explanation of key components / design choices

Stratified val split: preserves class proportions (important with imbalanced datasets).

Class weights in CrossEntropyLoss: inverse frequency weighting reduces bias toward the majority class.

Monitoring AUC (on raw logits): AUC on continuous scores is more informative than argmax accuracy for imbalanced problems and useful for threshold choice.

Early stopping on val AUC: prevents overfitting and saves the checkpoint with the best validation discrimination performance.

Gradient clipping: stabilizes training (helps when fine-tuning large transformer models).

Visualizations: combined ROC+PR with Youden's J (optimal ROC point) and sample predictions provide qualitative and quantitative diagnostics.

How to use the saved model for inference (simple snippet)
import torch, timm
from torchvision import transforms
from PIL import Image
import numpy as np

MODEL_PATH = '/kaggle/working/best_deit_model_fixed.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model architecture
model = timm.create_model('deit_base_patch16_224', pretrained=False)
model.reset_classifier(num_classes=2, global_pool='avg')  # or apply same head code as training
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open('some_xray.png').convert('RGB')
inp = transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    out = model(inp)
    logits = out.cpu().numpy()
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps[:, 1] / np.sum(exps, axis=1)
    pred = int(np.argmax(logits, axis=1)[0])
    print("prob of PNEUMONIA:", float(probs[0]), "pred label idx:", pred)

Tuning / next steps (suggestions)

Try different learning rates and optimizers (AdamW, SGD + momentum).

Experiment with more aggressive augmentations, dropout, or label smoothing to reduce overfitting.

Use mixup/cutmix if appropriate for medical images (careful with label semantics).

Run k-fold cross-validation for better robustness estimates.

Calibrate output probabilities (Platt scaling / isotonic) if using for clinical thresholds.

Explore interpretation: Grad-CAM / attention maps to inspect where the model focuses.

Troubleshooting

Out of memory: lower BATCH_SIZE, reduce num_workers, or use torch.cuda.amp for mixed precision.

AUC==1 or suspiciously perfect results on val: check for data leakage (e.g., same images in train/val/test, inadvertent preprocessing differences), and verify stratified split is performed on orig_train_folder.

Slow CPU training: ensure script runs on GPU. On Kaggle, set accelerator or use GPU runtime.

Reproducibility

Seed is fixed (SEED = 42) and torch.backends.cudnn.deterministic = True is set to reduce nondeterminism. Note that exact bit-for-bit reproducibility across different hardware/torch/cuda versions may still vary.

File structure (recommended)
.
├─ train_deit.py            # your main training script / notebook
├─ requirements.txt         # optional
├─ README.md                # this file
└─ outputs/
   ├─ best_deit_model_fixed.pth
   ├─ figs/
   └─ test_metrics_fixed_full.json

License & attribution

This project uses the deit / timm implementation and public datasets (ensure dataset usage follows its license). Add your preferred license file (MIT / Apache) to the repo if you intend to distribute code.

Contact / Notes

If you want, I can:

generate a concise requirements.txt for this run,

produce a short inference script that wraps the saved model into a function or REST endpoint,

or convert this README into a polished GitHub README with badges and a short demo GIF.
