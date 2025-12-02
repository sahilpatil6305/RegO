import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np



class ASRProcessor:
    """
    Wrapper for Automatic Speech Recognition using HuggingFace Transformers.
    Uses a lightweight model (e.g. 'openai/whisper-tiny' or 'facebook/wav2vec2-base-960h') 
    to transcribe audio to text.
    """
    def __init__(self, model_name="openai/whisper-tiny", device="cpu"):
        self.device = device
        try:
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model=model_name, 
                device=0 if device == "cuda" else -1
            )
        except Exception as e:
            print(f"[ASR] Warning: Could not load ASR model {model_name}. Error: {e}")
            self.asr_pipeline = None

    def transcribe(self, raw_audio, sample_rate=16000):
        """
        Transcribe a batch of raw audio waveforms.
        Args:
            raw_audio: Tensor of shape (batch, channels, time) or (batch, time)
            sample_rate: Sampling rate of the audio
        Returns:
            List of strings (transcriptions)
        """
        if self.asr_pipeline is None:
            return [""] * raw_audio.shape[0]

        transcriptions = []
        raw_audio_np = raw_audio.cpu().numpy()
        
        for i in range(raw_audio_np.shape[0]):
            waveform = raw_audio_np[i]
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)
            
            try:
                result = self.asr_pipeline(waveform, generate_kwargs={"language": "english"})
                text = result.get("text", "")
                transcriptions.append(text)
            except Exception as e:
                transcriptions.append("")
        
        return transcriptions

class TextFeatureExtractor(nn.Module):
    """
    Extracts semantic embeddings from text using a pre-trained language model.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
    def forward(self, texts):
        """
        Args:
            texts: List of strings
        Returns:
            Tensor of shape (batch, embedding_dim)
        """
        if not texts:
            return None
            
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

class OptimalTransportLoss(nn.Module):
    """
    Advanced Optimal Transport Loss using Sinkhorn algorithm for cross-modal alignment.
    Aligns audio embeddings to text embeddings using entropic regularization.
    
    Hyperparameters:
        reg: Sinkhorn regularization (default: 0.05)
        max_iter: Maximum Sinkhorn iterations (default: 50)
        drift_threshold: Threshold to flag region drift (default: 0.2)
    """
    def __init__(self, reg=0.05, max_iter=50, drift_threshold=0.2):
        super().__init__()
        self.reg = reg
        self.max_iter = max_iter
        self.drift_threshold = drift_threshold
        
    def compute_cost_matrix(self, X, Y):
        """
        Compute pairwise squared Euclidean distance matrix.
        Args:
            X: Tensor of shape (n, D)
            Y: Tensor of shape (m, D)
        Returns:
            Cost matrix of shape (n, m)
        """
        # Efficient: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        X_sqnorms = (X ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        Y_sqnorms = (Y ** 2).sum(dim=1, keepdim=True)  # (m, 1)
        XY = torch.mm(X, Y.t())  # (n, m)
        cost = X_sqnorms + Y_sqnorms.t() - 2 * XY
        return torch.clamp(cost, min=0.0)
    
    def sinkhorn_algorithm(self, cost, reg=None, max_iter=None):
        """
        Sinkhorn algorithm for entropic regularized optimal transport.
        Args:
            cost: Cost matrix of shape (n, m)
            reg: Regularization parameter (default: self.reg)
            max_iter: Maximum iterations (default: self.max_iter)
        Returns:
            gamma: Transport plan (coupling matrix) of shape (n, m)
        """
        if reg is None:
            reg = self.reg
        if max_iter is None:
            max_iter = self.max_iter
            
        n, m = cost.shape
        
        # Initialize uniform distributions
        a = torch.ones(n, device=cost.device) / n
        b = torch.ones(m, device=cost.device) / m
        
        # Compute kernel K = exp(-cost / reg)
        # Clamp cost to avoid underflow/overflow in exp
        cost_stable = torch.clamp(cost, max=1000.0) 
        K = torch.exp(-cost_stable / reg)
        
        # Sinkhorn iterations
        u = torch.ones(n, device=cost.device)
        v = torch.ones(m, device=cost.device)
        
        for _ in range(max_iter):
            # Add epsilon to avoid division by zero
            u = a / (K @ v + 1e-9)
            v = b / (K.t() @ u + 1e-9)
        
        # Compute transport plan
        gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        # Check for NaNs in gamma and handle them
        if torch.isnan(gamma).any():
            # Fallback to uniform coupling if Sinkhorn fails
            gamma = torch.ones(n, m, device=cost.device) / (n * m)
            
        return gamma
    
    def sinkhorn_transport(self, X, Y, reg=None, max_iter=None):
        """
        Compute Sinkhorn transport between two sets of features.
        Args:
            X: Source features of shape (n, D)
            Y: Target features of shape (m, D)
            reg: Regularization parameter
            max_iter: Maximum iterations
        Returns:
            gamma: Transport plan of shape (n, m)
            ot_distance: Optimal transport distance (scalar)
        """
        # Check for empty inputs
        if X.shape[0] == 0 or Y.shape[0] == 0:
            return torch.zeros(X.shape[0], Y.shape[0], device=X.device), torch.tensor(0.0, device=X.device)

        cost = self.compute_cost_matrix(X, Y)
        gamma = self.sinkhorn_algorithm(cost, reg=reg, max_iter=max_iter)
        ot_distance = torch.sum(gamma * cost)
        return gamma, ot_distance
    
    def align_audio_to_text(self, audio_emb, text_emb):
        """
        Align audio embeddings to text embeddings using optimal transport.
        Args:
            audio_emb: Audio features of shape (B, D)
            text_emb: Text features of shape (B, D)
        Returns:
            audio_aligned: Aligned audio features of shape (B, D)
            ot_distance: Optimal transport distance (scalar)
        """
        gamma, ot_distance = self.sinkhorn_transport(audio_emb, text_emb)
        
        # Transport audio features toward text anchors:
        # audio_aligned_i = sum_j gamma_ij * text_emb_j / sum_j gamma_ij
        row_sums = gamma.sum(dim=1, keepdim=True) + 1e-9
        audio_aligned = (gamma @ text_emb) / row_sums
        
        return audio_aligned, ot_distance
    
    def region_wise_ot_drift(self, prev_region_feats, curr_region_feats):
        """
        Compute region-wise OT drift between previous and current features.
        Args:
            prev_region_feats: Previous region features of shape (N_prev, D_r)
            curr_region_feats: Current region features of shape (N_curr, D_r)
        Returns:
            drift_score: OT distance (scalar)
            gamma: Transport plan
        """
        gamma, drift_score = self.sinkhorn_transport(prev_region_feats, curr_region_feats)
        return drift_score, gamma
    
    def forward(self, audio_features, text_features):
        """
        Compute OT loss between audio and text features.
        Args:
            audio_features: Audio embeddings of shape (B, D)
            text_features: Text embeddings of shape (B, D)
        Returns:
            ot_loss: Optimal transport distance
        """
        _, ot_distance = self.sinkhorn_transport(audio_features, text_features)
        return ot_distance


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module that combines audio and text features using OT alignment.
    Supports weighted fusion of original audio, OT-aligned audio, and text features.
    
    Fusion formula:
        fused = w_audio * audio_features + w_aligned * audio_aligned + w_text * text_features
    
    Default weights: (0.5, 0.4, 0.1) for (audio, aligned, text)
    """
    def __init__(self, audio_dim=2048, text_dim=384, output_dim=2048, 
                 fusion_weights=(0.5, 0.4, 0.1)):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.fusion_weights = fusion_weights
        
        # Projection layers to align dimensions
        self.text_proj = nn.Linear(text_dim, audio_dim)
        self.fusion_proj = nn.Linear(audio_dim, output_dim)
        
        # Optional: learnable fusion weights
        self.learnable_weights = nn.Parameter(torch.tensor(fusion_weights))
        
    def fuse_audio_text(self, audio_features, audio_aligned, text_features, 
                        use_learnable_weights=False):
        """
        Fuse audio, aligned audio, and text features with weighted combination.
        Args:
            audio_features: Original audio features (B, D)
            audio_aligned: OT-aligned audio features (B, D)
            text_features: Text features (B, D) - already projected
            use_learnable_weights: Use learnable fusion weights
        Returns:
            fused_features: Fused features (B, output_dim)
        """
        if use_learnable_weights:
            weights = F.softmax(self.learnable_weights, dim=0)
            w_audio, w_aligned, w_text = weights[0], weights[1], weights[2]
        else:
            w_audio, w_aligned, w_text = self.fusion_weights
        
        # Weighted fusion
        fused = w_audio * audio_features + w_aligned * audio_aligned + w_text * text_features
        
        # Project to output dimension if needed
        if self.output_dim != self.audio_dim:
            fused = self.fusion_proj(fused)
        
        return fused
    
    def forward(self, audio_features, audio_aligned, text_features_raw):
        """
        Forward pass: project text and fuse with audio.
        Args:
            audio_features: Original audio features (B, audio_dim)
            audio_aligned: OT-aligned audio features (B, audio_dim)
            text_features_raw: Raw text features (B, text_dim)
        Returns:
            fused_features: Fused features (B, output_dim)
        """
        text_features_proj = self.text_proj(text_features_raw)
        fused = self.fuse_audio_text(audio_features, audio_aligned, text_features_proj)
        return fused


class RegionFeatureBuffer:
    """
    Buffer to store region-specific features for drift detection and replay.
    Maintains a sliding window of features for each region.
    """
    def __init__(self, buffer_size=1000, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.buffers = {}  # region_name -> list of tensors
        
    def update(self, region_name, features):
        """
        Update buffer for a specific region.
        Args:
            region_name: Name of the region
            features: Features tensor of shape (B, D_r)
        """
        if region_name not in self.buffers:
            self.buffers[region_name] = []
        
        self.buffers[region_name].append(features.detach().cpu())
        
        # Maintain buffer size
        if len(self.buffers[region_name]) > self.buffer_size:
            self.buffers[region_name] = self.buffers[region_name][-self.buffer_size:]
    
    def sample(self, region_name, n_samples=None):
        """
        Sample features from buffer for a specific region.
        Args:
            region_name: Name of the region
            n_samples: Number of samples to return (None = all)
        Returns:
            Sampled features tensor or None if buffer is empty
        """
        if region_name not in self.buffers or len(self.buffers[region_name]) == 0:
            return None
        
        all_feats = torch.cat(self.buffers[region_name], dim=0)
        
        if n_samples is None or n_samples >= all_feats.shape[0]:
            return all_feats.to(self.device)
        
        indices = torch.randperm(all_feats.shape[0])[:n_samples]
        return all_feats[indices].to(self.device)
    
    def get_all(self, region_name):
        """Get all features for a specific region."""
        return self.sample(region_name, n_samples=None)
    
    def save(self, filepath, experience_id):
        """Save region features to disk."""
        save_dict = {
            'experience_id': experience_id,
            'buffers': {k: torch.cat(v, dim=0) if v else torch.empty(0) 
                       for k, v in self.buffers.items()}
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load region features from disk."""
        save_dict = torch.load(filepath, map_location='cpu')
        self.buffers = {k: [v] for k, v in save_dict['buffers'].items() if v.numel() > 0}
        return save_dict['experience_id']
