import torch
import torch.nn as nn
from avalanche.training.plugins import SupervisedPlugin
from llm_guided_optimization import (
    ASRProcessor, TextFeatureExtractor, OptimalTransportLoss, 
    MultiModalFusion, RegionFeatureBuffer
)
import torch.nn.functional as F
import requests
import json
import re

class LLMOptimizationPlugin(SupervisedPlugin):
    """
    Avalanche Plugin to inject LLM-guided optimization (ASR + OT Loss) into the training loop.
    """
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        
        # Initialize components
        print("[LLM-Plugin] Initializing LLM Optimization components...")
        self.asr_processor = ASRProcessor(model_name=getattr(args, 'asr_model', "openai/whisper-tiny"), device=device)
        self.text_extractor = TextFeatureExtractor(device=device)
        self.ot_loss_fn = OptimalTransportLoss(
            reg=getattr(args, 'ot_reg', 0.05),
            max_iter=getattr(args, 'ot_iters', 50),
            drift_threshold=getattr(args, 'ot_drift_threshold', 0.2)
        )
        
        feat_dim = getattr(args, 'pretrain_feature_shape', 2048)
        self.fusion_module = MultiModalFusion(audio_dim=feat_dim, text_dim=384, output_dim=feat_dim).to(device)
        
        # Region feature buffer for drift detection
        buffer_size = getattr(args, 'region_buffer_size', 1000)
        self.region_buffer = RegionFeatureBuffer(buffer_size=buffer_size, device=device)
        
        # Track current experience for region buffer
        self.current_experience = 0
        
        self.captured_features = {}
        self.region_hooks = []  # Store hooks for region extraction

    def before_training_exp(self, strategy, **kwargs):
        """Setup hooks for region feature extraction before each experience."""
        model = strategy.model
        self.current_experience = strategy.experience.current_experience
        
        print(f"[LLM-Plugin] Setting up for experience {self.current_experience}")
        
        def get_features_hook(name):
            def hook(module, input, output):
                self.captured_features[name] = output.detach()
            return hook
        
        # Remove existing hooks
        for handle in self.region_hooks:
            handle.remove()
        self.region_hooks.clear()
        self.captured_features.clear()
        
        # Unwrap DataParallel/TupleWrapper
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        # Handle TupleWrapper (from train.py)
        if hasattr(model, 'model'):
            model = model.model
        
        # For ResNet50 in Sequential: model = Sequential(resnet, flatten, linear)
        if isinstance(model, nn.Sequential) and len(model) > 0:
            resnet = model[0]  # ResNet50 backbone
            
            # Register hooks on ResNet layers for region extraction
            if hasattr(resnet, 'layer2'):
                self.region_hooks.append(resnet.layer2.register_forward_hook(get_features_hook('early')))
            if hasattr(resnet, 'layer3'):
                self.region_hooks.append(resnet.layer3.register_forward_hook(get_features_hook('mid')))
            if hasattr(resnet, 'layer4'):
                self.region_hooks.append(resnet.layer4.register_forward_hook(get_features_hook('late')))
            
            # Also hook final features (before classifier)
            if len(model) >= 2:
                self.region_hooks.append(model[-2].register_forward_hook(get_features_hook('features')))
            
            print(f"[LLM-Plugin] Registered {len(self.region_hooks)} region hooks")
        else:
            print("[LLM-Plugin] Warning: Unexpected model structure, cannot register region hooks")

        # Add fusion module params to optimizer if not already there
        # This is a bit hacky but necessary if we introduce new trainable params
        fusion_param_ids = {id(p) for p in self.fusion_module.parameters()}
        found = False
        
        for group in strategy.optimizer.param_groups:
            for param in group['params']:
                if id(param) in fusion_param_ids:
                    found = True
                    break
            if found:
                break
        
        if not found:
            strategy.optimizer.add_param_group({'params': self.fusion_module.parameters()})
            print("[LLM-Plugin] Fusion module parameters added to optimizer.")

    def before_backward(self, strategy, **kwargs):
        """
        Compute LLM/OT loss with advanced cross-modal alignment and add it to strategy.loss.
        
        Flow:
        1. Extract raw audio from batch
        2. Transcribe audio to text (ASR)
        3. Extract text embeddings
        4. Capture audio features from model hook
        5. Align audio to text using Sinkhorn OT
        6. Fuse audio, aligned audio, and text features
        7. Compute semantic consistency loss + OT distance loss
        8. Add combined loss to strategy.loss
        """
        print("[LLM-Plugin] before_backward called", flush=True)
        
        # Unpack batch - Standard Avalanche format: (x, y, t)
        # x is now ((spec, raw), ...) due to custom collate
        mbatch = strategy.mbatch
        x_mb = mbatch[0]
        
        raw_audio = None
        if isinstance(x_mb, (tuple, list)) and len(x_mb) == 2:
            # x_mb is (spec, raw)
            raw_audio = x_mb[1]
        
        if raw_audio is None:
            print("[LLM-Plugin] No raw audio available in batch, skipping optimization")
            return

        try:
            # 1. ASR: Transcribe audio to text
            print("[LLM-Plugin] Starting ASR...", flush=True)
            transcripts = self.asr_processor.transcribe(raw_audio)
            print(f"[LLM-Plugin] ASR finished. Transcripts: {transcripts[:2]}...", flush=True)
            
            # 2. Text Embeddings: Extract semantic features from transcripts
            print("[LLM-Plugin] Extracting text features...", flush=True)
            text_feats = self.text_extractor(transcripts)  # (B, 384)
            print("[LLM-Plugin] Text features extracted.", flush=True)
            
            # 3. Audio Features: Captured from model hook
            audio_feats = self.captured_features.get('features')
            print(f"[LLM-Plugin] Audio features captured: {audio_feats is not None}", flush=True)
            
            # Debug logging to file
            with open("debug_log.txt", "a") as f:
                f.write(f"[LLM-Debug] raw_audio shape: {raw_audio.shape}\n")
                if text_feats is not None: f.write(f"[LLM-Debug] text_feats shape: {text_feats.shape}\n")
                if audio_feats is not None: f.write(f"[LLM-Debug] audio_feats shape: {audio_feats.shape}\n")
            
            if audio_feats is not None and text_feats is not None:
                # Project text features to audio dimension
                text_feats_proj = self.fusion_module.text_proj(text_feats)
                
                # 4. Cross-modal OT alignment: Align audio to text using Sinkhorn
                audio_aligned, ot_dist_crossmodal = self.ot_loss_fn.align_audio_to_text(
                    audio_feats, text_feats_proj
                )
                
                # 5. Multi-modal fusion
                fused_features = self.fusion_module.fuse_audio_text(
                    audio_feats, audio_aligned, text_feats_proj,
                    use_learnable_weights=False
                )
                
                # 6. Semantic Consistency Loss
                sim = F.cosine_similarity(audio_feats, text_feats_proj)
                semantic_loss = (1 - sim).mean()
                
                # 7. Cross-modal OT Loss
                ot_loss = ot_dist_crossmodal
                
                # 8. Region-wise OT Drift Penalty
                region_ot_penalty = torch.tensor(0.0, device=self.device)
                region_drift_scores = {}
                
                for region_name in ['early', 'mid', 'late']:
                    curr_feats = self.captured_features.get(region_name)
                    if curr_feats is None:
                        continue
                    
                    # Flatten spatial dimensions if needed (e.g., conv output)
                    if curr_feats.ndim > 2:
                        curr_feats = curr_feats.mean(dim=[2, 3])  # Global avg pool
                    
                    # Get previous experience features
                    prev_feats = self.region_buffer.sample(region_name, n_samples=min(100, curr_feats.shape[0]))
                    
                    if prev_feats is not None and prev_feats.shape[0] > 0:
                        # Ensure same dimension
                        if prev_feats.shape[1] != curr_feats.shape[1]:
                            continue
                        
                # 9. Combined loss
                sem_weight = getattr(self.args, 'semantic_loss_weight', 0.1)
                ot_weight = getattr(self.args, 'ot_loss_weight', 0.1)
                region_weight = getattr(self.args, 'region_ot_weight', 0.05)
                
                extra_loss = (sem_weight * semantic_loss + 
                             ot_weight * ot_loss + 
                             region_weight * region_ot_penalty)
                
                # Check for NaNs
                if torch.isnan(extra_loss) or torch.isinf(extra_loss):
                    print(f"[LLM-Plugin] Warning: NaN/Inf detected in loss! Skipping update.")
                    print(f"  Sem: {semantic_loss.item()}, OT: {ot_loss.item()}, Region: {region_ot_penalty.item()}")
                    # Zero out loss to prevent contamination
                    extra_loss = torch.tensor(0.0, device=self.device)
                else:
                    strategy.loss += extra_loss
                
                # Log
                with open("debug_log.txt", "a") as f:
                    f.write(f"[LLM] Loss: {extra_loss.item():.4f} ")
                    f.write(f"(Sem:{semantic_loss.item():.3f}, OT:{ot_loss.item():.3f}, ")
                    f.write(f"Region:{region_ot_penalty.item():.3f})\n")
                    if region_drift_scores:
                        f.write(f"[LLM] Region Drift: {region_drift_scores}\n")
                        
                # 10. Update Region Buffer (only if loss was valid)
                if extra_loss > 0:
                    for region_name in ['early', 'mid', 'late']:
                        feats = self.captured_features.get(region_name)
                        if feats is not None:
                            if feats.ndim > 2:
                                feats = feats.mean(dim=[2, 3])
                            self.region_buffer.update(region_name, feats)
                
        except Exception as e:
            print(f"[LLM-Plugin] Error in before_backward: {e}")
            import traceback
            traceback.print_exc()
            
    def after_training_exp(self, strategy, **kwargs):
        """
        At the end of each experience:
        1. Summarize performance (accuracy, loss, drift).
        2. Query LLM for modulation parameters.
        3. Update hyperparameters for next experience.
        """
        print(f"[LLM-Plugin] Finishing experience {self.current_experience}")
        
        # 1. Gather Metrics
        # Strategy.evaluator.get_last_metrics() might be empty if eval didn't run
        # We'll use the tracked losses from before_backward if available, or just general status
        
        # Calculate average drift for this experience
        avg_drift = {}
        if hasattr(self, 'region_buffer'):
            # This is a bit rough, we should ideally track this during training
            pass
            
        summary = {
            "experience": self.current_experience,
            "status": "completed",
            # "avg_loss": ... # TODO: track this better
        }
        
        # 2. Construct Prompt
        prompt = self._construct_llm_prompt(summary)
        
        # 3. Query LLM
        print("[LLM-Plugin] Querying LLM for modulation parameters...")
        response = self._query_llm(prompt)
        
        if response:
            print(f"[LLM-Plugin] LLM Response: {response}")
            # 4. Parse and Apply Modulation
            self._apply_modulation(response, strategy)
        else:
            print("[LLM-Plugin] Failed to get LLM response, keeping current parameters.")

    def _construct_llm_prompt(self, summary):
        return f"""
        You are an AI optimization assistant for a Continual Learning model (RegO).
        The model just finished Experience {summary['experience']}.
        
        Current Hyperparameters:
        - Learning Rate: {getattr(self.args, 'lr', 'unknown')}
        - OT Regularization: {self.ot_loss_fn.reg}
        - OT Loss Weight: {getattr(self.args, 'ot_loss_weight', 0.1)}
        - Region OT Weight: {getattr(self.args, 'region_ot_weight', 0.05)}
        
        Task: Analyze the training state and suggest updates for the next experience to balance plasticity (learning new) and stability (remembering old).
        
        Return a JSON object with these keys:
        - "learning_rate_factor": float (multiplier for LR, e.g., 0.9 to decrease, 1.1 to increase)
        - "ot_reg": float (new optimal transport regularization, range 0.01-0.1)
        - "ot_loss_weight": float (weight for OT loss, range 0.0-1.0)
        - "region_ot_weight": float (weight for region drift penalty, range 0.0-0.5)
        - "reasoning": string (brief explanation)
        
        Respond ONLY with the JSON string.
        """

    def _query_llm(self, prompt):
        url = "http://localhost:11434/api/generate"
        model = getattr(self.args, 'llm_model', 'llama3.2')
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            else:
                print(f"[LLM-Plugin] Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"[LLM-Plugin] Connection error: {e}")
            return None

    def _apply_modulation(self, response_text, strategy):
        try:
            # Clean response to ensure valid JSON
            json_str = response_text.strip()
            # If wrapped in markdown code blocks, remove them
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            
            params = json.loads(json_str)
            
            # Apply updates
            if 'ot_reg' in params:
                self.ot_loss_fn.reg = float(params['ot_reg'])
                print(f"[LLM-Plugin] Updated OT Regularization to {self.ot_loss_fn.reg}")
                
            if 'ot_loss_weight' in params:
                self.args.ot_loss_weight = float(params['ot_loss_weight'])
                print(f"[LLM-Plugin] Updated OT Loss Weight to {self.args.ot_loss_weight}")
                
            if 'region_ot_weight' in params:
                self.args.region_ot_weight = float(params['region_ot_weight'])
                print(f"[LLM-Plugin] Updated Region OT Weight to {self.args.region_ot_weight}")
                
            if 'learning_rate_factor' in params:
                factor = float(params['learning_rate_factor'])
                for param_group in strategy.optimizer.param_groups:
                    param_group['lr'] *= factor
                print(f"[LLM-Plugin] Adjusted Learning Rate by factor {factor}")
                
            print(f"[LLM-Plugin] Modulation applied based on reasoning: {params.get('reasoning', 'None')}")
            
        except json.JSONDecodeError:
            print(f"[LLM-Plugin] Failed to parse JSON response: {response_text}")
        except Exception as e:
            print(f"[LLM-Plugin] Error applying modulation: {e}")
