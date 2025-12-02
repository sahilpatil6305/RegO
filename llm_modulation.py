# llm_modulation.py
import json
import requests
import torch
from typing import Dict, Any, Optional


class LLMGuidedModulator:
    """
    Region-wise LR / WD modulation driven by local Ollama LLM.
    - call_llm_for_modulation: queries ollama local server and expects pure JSON output.
    - apply_modulation: rebuilds optimizer.param_groups using region multipliers.
    """

    def __init__(self, model: torch.nn.Module, base_lr: float, base_wd: float,
                 ollama_url: str = "http://localhost:11434/api/generate",
                 ollama_model: str = "llama3"):
        self.model = model
        self.base_lr = float(base_lr)
        self.base_wd = float(base_wd)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Region mapping tuned for nn.Sequential(torchvision.models.resnet..., AvgPool, Flatten)
        # Parameter names typically look like: "0.layer1.0.conv1.weight"
        self.region_map = {
            "early": ["0.conv1", "0.bn1", "0.layer1", "0.layer2"],
            "mid": ["0.layer3"],
            "late": ["0.layer4", "0.fc", "classifier"]
        }

    # -------------------------
    # Build a concise summary for LLM
    # -------------------------
    def build_summary(self, exp_id: int, train_metrics: Dict[str, Any],
                      prev_test_metrics: Optional[Dict[str, Any]] = None) -> str:
        prev = prev_test_metrics or {}

        def get_val(d, k):
            return d.get(k, "N/A")

        prompt = f"""
You are given a short summary of a continual-learning experience for an audio deepfake detection model.

Experience: {exp_id}

TRAIN METRICS:
  Top1_Acc_Exp/train_stream = {get_val(train_metrics, 'Top1_Acc_Exp/train_stream')}
  Loss_Exp/train_stream     = {get_val(train_metrics, 'Loss_Exp/train_stream')}

PREVIOUS TEST METRICS:
  Top1_Acc_Exp/test_stream  = {get_val(prev, 'Top1_Acc_Exp/test_stream')}
  Loss_Exp/test_stream      = {get_val(prev, 'Loss_Exp/test_stream')}
  Forgetting_Exp/test_stream = {get_val(prev, 'Forgetting_Exp/test_stream')}

GOAL:
  - Reduce forgetting on previous experiences.
  - Improve adaptation to new fake types while avoiding overfitting.

Return a JSON object ONLY, with keys "early","mid","late", each mapping to:
  {{"lr_mult": float_between_0.1_and_2.0, "wd_mult": float_between_0.1_and_2.0}}

Example output:
{{ "early": {{"lr_mult": 0.8, "wd_mult": 1.1}}, "mid": {{"lr_mult": 1.0, "wd_mult": 1.0}}, "late": {{"lr_mult": 1.3, "wd_mult": 0.9}} }}

DO NOT output any text outside of the JSON.
"""
        return prompt

    # -------------------------
    # Call Ollama; expect JSON in the model's textual output
    # -------------------------
    def call_llm_for_modulation(self, summary_text: str) -> Dict[str, Dict[str, float]]:
        payload = {
            "model": self.ollama_model,
            "prompt": summary_text,
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False
        }

        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # Ollama response format may vary; try common keys
            # Prioritize .get("response") or .get("text"), else fallback to entire 'data' as string
            text = None
            if isinstance(data, dict):
                text = data.get("response") or data.get("text") or data.get("output") or data.get("result")
            if text is None:
                # try flattening some known structure
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text")
            if text is None:
                # last resort: stringify whole response
                text = json.dumps(data)

            # The model should output JSON only: parse it
            modulation = json.loads(text)
            # Basic validation: ensure keys present
            for r in ["early", "mid", "late"]:
                if r not in modulation:
                    raise ValueError(f"Missing region in modulation: {r}")

            return modulation

        except Exception as e:
            print("[LLM][ERROR] Ollama call failed:", e)
            print("[LLM] Falling back to identity multipliers.")
            return {
                "early": {"lr_mult": 1.0, "wd_mult": 1.0},
                "mid": {"lr_mult": 1.0, "wd_mult": 1.0},
                "late": {"lr_mult": 1.0, "wd_mult": 1.0},
            }

    # -------------------------
    # Apply modulation to optimizer param groups
    # -------------------------
    def apply_modulation(self, optimizer: torch.optim.Optimizer, modulation_dict: Dict[str, Dict[str, float]]):
        def which_region(param_name: str) -> str:
            for region, subs in self.region_map.items():
                if any(s in param_name for s in subs):
                    return region
            return "mid"

        new_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            region = which_region(name)
            cfg = modulation_dict.get(region, {"lr_mult": 1.0, "wd_mult": 1.0})
            lr = float(self.base_lr * float(cfg.get("lr_mult", 1.0)))
            wd = float(self.base_wd * float(cfg.get("wd_mult", 1.0)))
            new_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

        optimizer.param_groups.clear()
        optimizer.param_groups.extend(new_groups)
