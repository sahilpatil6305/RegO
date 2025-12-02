## RegO: Region-wise Optimal Transport for Audio Deepfake Detection

## Overview

**RegO** is a novel continual learning framework for audio deepfake detection that combines cutting-edge techniques to achieve state-of-the-art performance in identifying synthetic audio.

### Key Innovations

1. **Region-wise Optimal Transport (OT) Alignment** - Cross-modal feature alignment between audio and text embeddings
2. **LLM-Guided Optimization** - Adaptive hyperparameter tuning using Large Language Models
3. **Multi-Modal Fusion** - Combines audio spectrograms and text transcriptions
4. **Continual Learning** - Prevents catastrophic forgetting across multiple datasets

---

## Architecture

```
Audio Input → Wav2Vec2 Encoder → Audio Features ──┐
                                                   ├─→ OT Alignment → Fusion → Classification
Text Input → BERT Encoder → Text Features ────────┘
                                ↑
                          LLM Optimization
```

### Components

- **Audio Encoder:** Wav2Vec2 (pre-trained on speech)
- **Text Encoder:** BERT (for transcription embeddings)
- **OT Module:** Sinkhorn algorithm for cross-modal alignment
- **LLM Plugin:** Ollama-based adaptive optimization
- **Fusion:** Learnable weighted combination
- **Classifier:** Binary (real/fake) with continual learning

---


### Installation

```bash
# Clone the repository
cd RegO

# Create virtual environment
python -m venv rego-env

# Activate environment
# Windows:
.\rego-env\Scripts\Activate.ps1
# Linux/Mac:
source rego-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Place your audio datasets in the following structure:
```
dataset/
├── dataset1/
│   ├── train/
│   └── test/
├── dataset2/
│   ├── train/
│   └── test/
└── dataset3/
    ├── train/
    └── test/
```

---


### Training

```bash
# Activate environment
.\rego-env\Scripts\Activate.ps1

# Train with default configuration
python train.py

# Train with custom config
python train.py --config yaml/clear10/train.yaml
```


## Configuration

Main configuration file: `yaml/clear10/train.yaml`

### Ablation Studies

Pre-configured ablation experiments:
- `yaml/ablation/exp1_baseline.yaml` - Baseline without OT/LLM
- `yaml/ablation/exp2_no_llm.yaml` - OT only, no LLM

---

## Project Structure

```
RegO/
├── train.py                    # Main training script
├── run_eval.py                 # Evaluation script
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── RUN_GUIDE.md               # Detailed usage guide
│
├── data/                       # Data loading modules
│   ├── load_dataset.py
│   ├── collate_fn.py
│   └── parse_data_path.py
│
├── models/                     # Model architectures
│   └── self_supervised_models/
│       └── moco_v2.py
│
├── plugins/                    # Optimization plugins
│   ├── llm_optimization_plugin.py
│   └── gradient_clipping_plugin.py
│
├── metrics/                    # Evaluation metrics
│   ├── custom_metrics.py
│   └── get_metric_all.py
│
└── yaml/                       # Configuration files
    ├── clear10/
    └── ablation/
```

---

## Technical Details

### Optimal Transport Alignment

The OT module uses the Sinkhorn algorithm to compute optimal transport between audio and text feature distributions:

```python
# Compute cost matrix
C = torch.cdist(audio_features, text_features)

# Sinkhorn iterations
transport_plan = sinkhorn(C, reg=0.1, num_iters=100)

# Align features
aligned_audio = transport_plan @ text_features
```

### LLM-Guided Optimization

The LLM plugin analyzes training metrics and suggests hyperparameter adjustments:

1. Collect training statistics (loss, accuracy, gradients)
2. Query LLM with performance summary
3. Parse LLM suggestions (learning rate, regularization)
4. Apply modulation to optimizer

### Continual Learning

Uses Avalanche framework with:
- **Naive strategy** - Sequential training on multiple datasets
- **Experience replay** - Prevents catastrophic forgetting
- **Metrics tracking** - Accuracy, forgetting, forward transfer

---

## Troubleshooting

### Common Issues

**1. LLM 404 Error**
```
[LLM-Plugin] Error: 404 - {"error":"model 'llama3.2' not found"}
```
**Solution:** Install Ollama and pull the model:
```bash
ollama pull llama3.2
```
Or disable LLM in config: `llm_optimization.enable: False`

```

**3. Import Errors**
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

## Dependencies

Key libraries:
- **PyTorch** (2.0+) - Deep learning framework
- **Transformers** (4.30+) - BERT, Wav2Vec2 models
- **Avalanche** (0.3+) - Continual learning
- **POT** (0.9+) - Optimal transport
- **Librosa** (0.10+) - Audio processing
- **Whisper** (OpenAI) - Speech recognition

See `requirements.txt` for complete list.

---

## License

This project is for academic research purposes.

---

## Authors

- **Sahil Patil** 

---

## Acknowledgments

- Avalanche continual learning framework
- Hugging Face Transformers
- OpenAI Whisper
- Python Optimal Transport library



## Related Work

- [Wav2Vec2](https://arxiv.org/abs/2006.11477) - Self-supervised speech representations
- [Optimal Transport](https://arxiv.org/abs/1803.00567) - Computational optimal transport
- [Avalanche](https://avalanche.continualai.org/) - Continual learning library
