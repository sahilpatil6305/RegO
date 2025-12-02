import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from get_config import get_config
from load_dataset import get_data_set_offline
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, confusion_matrix_metrics
from tools.utils import set_seed, build_logger
from torchvision.models import ResNet50_Weights
import torchvision
from collate_fn import CollatorAudio
from custom_metrics import PrecisionRecallF1Metrics

def main():
    # Load configuration
    args = get_config()
    args.data_folder_path = str(Path(args.data_folder_path).resolve())
    
    # Set seed
    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model (ResNet50)
    print("[INFO] Initializing ResNet50 for Audio Spectrograms...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(2048, args.num_classes)
    
    # Wrap model to handle (spectrogram, raw_audio) tuple
    class TupleWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                return self.model(x[0])
            return self.model(x)
    
    model = TupleWrapper(model)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)

    # Load Saved Model
    # Assuming Experience 0
    model_path = f'../{args.split}/model/model_{args.method}_offline_time00.pth'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    print(f"[INFO] Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[WARN] Missing keys: {missing[:5]}...")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}...")

    # Load Dataset
    print("[INFO] Loading dataset...")
    scenario = get_data_set_offline(args)

    # Setup Evaluation Plugin
    text_logger, interactive_logger, eval_plugin = build_logger(args.split)
    eval_plugin.metrics.append(PrecisionRecallF1Metrics())
    eval_plugin.metrics.append(confusion_matrix_metrics(num_classes=args.num_classes, save_image=False, normalize='true'))

    # Custom Strategy for Collate Fn
    custom_collate = CollatorAudio()
    
    class CustomNaive(Naive):
        def make_eval_dataloader(self, num_workers=0, pin_memory=True, **kwargs):
            self.dataloader = DataLoader(
                self.adapted_dataset,
                batch_size=self.eval_mb_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate
            )

    # Initialize Strategy
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    cl_strategy = CustomNaive(
        model=model,
        optimizer=optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size=args.batch_size,
        train_epochs=args.nepoch,
        eval_mb_size=args.batch_size,
        evaluator=eval_plugin,
        device=device
    )

    # Run Evaluation
    print("\n" + "="*80)
    print("[STARTING EVALUATION]")
    print("="*80)

    # Filter valid experiences
    valid_test_experiences = []
    for exp in scenario.test_stream:
        if len(exp.dataset) > 0:
            valid_test_experiences.append(exp)
    
    if len(valid_test_experiences) > 0:
        print(f"Found {len(valid_test_experiences)} valid test experiences.")
        test_res = cl_strategy.eval(valid_test_experiences)
        print("\n[EVALUATION RESULTS]")
        print(json.dumps(test_res, indent=4, default=str))
        
        # Save metrics
        os.makedirs(f"../{args.split}/metric/", exist_ok=True)
        with open(f"../{args.split}/metric/test_metric_eval_only.json", "w") as out_file:
            json.dump(test_res, out_file, indent=6, default=str)
        print(f"[SAVED] Metrics saved to ../{args.split}/metric/test_metric_eval_only.json")
    else:
        print("[WARN] No valid test experiences found.")

if __name__ == "__main__":
    main()
