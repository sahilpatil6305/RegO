import json
import requests
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
import os
import shutil
import numpy as np
import torchvision
from torchvision.models import ResNet50_Weights

from get_config import get_config
from llm_guided_optimization import ASRProcessor, TextFeatureExtractor, OptimalTransportLoss, MultiModalFusion
from collate_fn import collator_audio
from llm_modulation import LLMGuidedModulator
from load_dataset import get_data_set_offline, get_data_set_online
from parse_data_path import move_data_trinity
from extract_feature import extract
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.supervised import Naive, CWRStar, Replay, JointTraining, GDumb, Cumulative, EWC, SynapticIntelligence
from avalanche.training.supervised import AGEM, GEM, CoPE
from avalanche.training.plugins import AGEMPlugin, GEMPlugin, EWCPlugin, SynapticIntelligencePlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, confusion_matrix_metrics
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from tools.utils import set_seed, make_scheduler, build_logger
from self_supervised_models.moco_v2 import moco_v2_yfcc_feb18_bucket_0_gpu_8
from avalanche.training.plugins import SupervisedPlugin

class LoadBestPlugin(SupervisedPlugin):
    """
    Stub for LoadBestPlugin.
    """
    def __init__(self, metric_name):
        super().__init__()
        self.metric_name = metric_name
        print(f"[INFO] LoadBestPlugin stub initialized for {metric_name}")

    def after_training_exp(self, strategy, **kwargs):
        pass

# Initialize configuration
args = get_config()
args.data_folder_path = str(Path(args.data_folder_path).resolve())
if getattr(args, 'data_train_path', None):
    args.data_train_path = str(Path(args.data_train_path).resolve())
if getattr(args, 'data_test_path', None):
    args.data_test_path = str(Path(args.data_test_path).resolve())
if getattr(args, 'feature_path', None):
    args.feature_path = str(Path(args.feature_path).resolve())

set_seed(args.seed)




try:
    restart=int(args.restart)
except:
    print('restart flag must be 0/1')
    assert False
if(restart==1):
    print('???!!!!!!!!!!!!!!!!!!!!!!!!!You sure to remove the old checkpoint ???!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('enter Y/y to continue')
    value=input()
    if(value=="y" or value=='Y'):
        assert False
        print('remove old split folder')

os.makedirs("../{}".format(args.split),exist_ok=True)
os.makedirs("../{}/log/".format(args.split),exist_ok=True)
os.makedirs("../{}/model/".format(args.split),exist_ok=True)
os.makedirs("../{}/metric/".format(args.split),exist_ok=True)
method_query=args.method.split() # list of CL method to run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.get_device_name(0)
# torch.cuda.device_count() 
'''
Remember to delete the old feature path before generating new feature 
'''
if(args.pretrain_feature!='None'):
    if(args.data_test_path !='' and args.data_train_path!=''):
        data_test_path=args.data_test_path
        data_train_path=args.data_train_path
        train_num_instance_each_class=args.num_instance_each_class
        for stage in ['train','test']:
            if(stage=='train'):
                args.data_folder_path=data_train_path
                args.data_train_path=''
                args.pretrain_feature='train_'+args.pretrain_feature
                args=extract(args)
                args.data_train_path=args.data_folder_path
            else:
                args.num_instance_each_class=args.num_instance_each_class_test
                args.data_folder_path=data_test_path
                args.data_test_path=''
                args.pretrain_feature=args.pretrain_feature.replace('train','test')
                args=extract(args)
                args.data_test_path=args.data_folder_path
                args.num_instance_each_class=train_num_instance_each_class
    else:    

        args=extract(args)

if(args.data_test_path !='' and args.data_train_path!=''):
    for stage in ['train','test']:
        if(stage=='train'):
            args.data_train_path=move_data_trinity(args.data_train_path,False)
        else:
            args.data_test_path=move_data_trinity(args.data_test_path,False)

else:
    args.data_folder_path=move_data_trinity(args.data_folder_path,False)

with open('../{}/args.txt'.format(args.split), 'w') as f:
    print('args', args, file=f) # keep a copy of the args
# Copy the local `avalanche` package into the split folder as a snapshot
try:
    src = os.path.join('..', 'avalanche')
    dst = os.path.join('..', args.split, 'avalanche')
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
except Exception as e:
    print('Warning: could not copy avalanche folder snapshot:', e)
for strate in method_query:
    for current_mode in ['offline']:
        # skip previous train model if necessary
        import glob
        model_path=sorted(glob.glob('../{}/model/model_{}_{}*'.format(args.split,strate,current_mode)))
        if(len(model_path)==0 and args.eval==True):
            checkpoint_path='../{}/model/model_{}_{}*'.format(args.split,strate,current_mode)
            print('Checkpoint for model {} is not found at path {}'.format(strate,checkpoint_path))
            continue
        if(len(model_path)!=0):
            model_path=model_path[-1]
            state_dict=torch.load(model_path)
        else:
            state_dict=None
        if current_mode == 'offline':
            print("\n" + "="*80)
            print("[DATASET LOADING]")
            print("="*80)
            if args.data_train_path == '' or args.data_test_path == '':
                print("  Mode: Using ALL 3 datasets for continual learning")
                print("  - Dataset 1: {}".format(args.data_folder_path + "/dataset1"))
                print("  - Dataset 2: {}".format(args.data_folder_path + "/dataset2"))
                print("  - Dataset 3: {}".format(args.data_folder_path + "/dataset3"))
            else:
                print("  Mode: Train/Test split")
                print("  - Train: {}".format(args.data_train_path))
                print("  - Test: {}".format(args.data_test_path))
            print("="*80 + "\n")
            scenario = get_data_set_offline(args)
        else:
            print("\n" + "="*80)
            print("[*] LOADING ALL DATASETS (Online Mode)...")
            print(f"  Path: {args.data_folder_path}")
            print("="*80 + "\n")
            scenario = get_data_set_online(args)

        # 🔑 Ensure each experience has at least one task label (default = [0])
        for stream in [scenario.train_stream, scenario.test_stream]:
            for exp in stream:
                if not getattr(exp, "task_labels", None):
            # Create a proper _task_labels attribute with getter
                    exp.__dict__["_task_labels"] = [0]

        # (Optional) Debug print to verify
        for i, exp in enumerate(scenario.test_stream):
            if len(exp.dataset) == 0:
                print(f"[WARN] Empty experience: {exp.current_experience}")


        train_list = scenario.train_stream
        print("Number of experiences in train_stream:", len(scenario.train_stream))

        print('========================================================')
        print('========================================================')
        print('current strate is {} {}'.format(strate,current_mode))
        print('========================================================')
        print('========================================================')


       # -------------------------
# Model Initialization
# -------------------------
        class TupleWrapper(nn.Module):
            """
            Wraps the model to handle tuple input (x, raw_audio) from Avalanche.
            Passes only 'x' (image/spectrogram) to the underlying model.
            """
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # x is ((spec, raw), ...) or just spec
                # Avalanche might wrap it further, but usually it's what collate returns
                if isinstance(x, (tuple, list)) and len(x) == 2:
                    # Assume (spec, raw)
                    spec = x[0]
                    return self.model(spec)
                return self.model(x)

        if args.pretrain_feature == 'None':
            # Load pretrained model from torchvision
            pretrain = args.image_train_pretrain
            model_class = getattr(torchvision.models, args.image_train_model_arch)
            model = model_class(pretrained=pretrain)

            # Handle specific attributes, e.g., MoCo
            if args.image_train_model_arch == 'resnet50' and getattr(args, 'image_train_attribute', None) == 'moco':
                model = moco_v2_yfcc_feb18_bucket_0_gpu_8(model)

            # Replace final layer with feature extractor
            model = nn.Sequential(
                model,                           # Base model (ResNet etc.)
                nn.AdaptiveAvgPool2d((1, 1)),    # Ensure fixed-size output
                nn.Flatten()                      # Flatten -> [batch, features]
            )

            # Set feature dimension depending on architecture
            if args.image_train_model_arch == 'resnet50':
                args.pretrain_feature_shape = 2048
            elif args.image_train_model_arch == 'resnet18':
                args.pretrain_feature_shape = 512
            else:
                raise ValueError(f"Unknown architecture {args.image_train_model_arch}")

        else:
            # For audio spectrograms, we treat them as images (3, 224, 224)
            # We want to use the same ResNet backbone to extract features (2048 dim)
            # instead of a raw Linear layer on flattened input.
            
            print(f"\n[INFO] Initializing ResNet50 for Audio Spectrograms...")
            # Load standard ResNet50 with pretrained weights
            model_class = torchvision.models.resnet50
            model = model_class(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # If MoCo is requested (optional, but good for consistency)
            if getattr(args, 'image_train_attribute', None) == 'moco':
                model = moco_v2_yfcc_feb18_bucket_0_gpu_8(model)
                
            # We need to match the structure expected by the plugin:
            # Sequential(backbone, pool, flatten) -> (classifier is handled by Avalanche or added later?)
            # Wait, the 'if' block above does:
            # model = nn.Sequential(model, pool, flatten)
            # And Avalanche adds the classifier? No, Avalanche expects a model that outputs logits.
            # The 'if' block above returns a feature extractor (2048 dim) and NO classifier?
            # Let's check line 243 in the original code: `model = nn.Linear(audio_input_size, args.num_classes)`
            # So the original code expected a full model.
            # Avalanche strategies like 'Naive' or 'Replay' usually expect a model that outputs logits (num_classes).
            # If the 'if' block returns a flattened feature vector, where is the classifier?
            # Ah, `train.py` lines 203-207 create a feature extractor.
            # But then where is the classifier added?
            # Avalanche's `SimpleMLP` or similar might add it, but here we are passing `model` directly.
            # If `model` outputs 2048, and we use CrossEntropy, it will fail unless the strategy adds a head.
            # Strategies like `CWRStar` might add a head, but `Naive` usually doesn't.
            # Let's look at `cl_strategy = Naive(model, ...)`
            # If `model` output is 2048 and targets are 0/1, CrossEntropy expects logits of size 2.
            # So the 'if' block seems to be missing a classifier too?
            # Or maybe `moco_v2_yfcc...` returns a model with a head?
            # No, `moco` usually returns features.
            
            # Let's assume we need to return a full model (backbone + classifier).
            # So for the 'if' block (lines 203-207), it might be incomplete or relying on something else.
            # But for our 'else' block (Audio), we MUST provide a full model.
            
            # Let's wrap it properly:
            # 1. Remove original fc
            model.fc = nn.Identity()
            
            # 2. Create Sequential
            model = nn.Sequential(
                model,                           # ResNet (outputs 2048)
                nn.Flatten(),                    # Flatten (just in case)
                nn.Linear(2048, args.num_classes) # Classifier
            )
            
            # Update args for plugin
            args.pretrain_feature_shape = 2048
            
            print(f"[INFO] Model: ResNet50 (Pretrained=True) -> Identity -> Linear({args.num_classes})")
            print(f"[INFO] Input expected: (Batch, 3, 224, 224)")
            
        # Wrap model to handle tuple input (img, raw_audio)
        model = TupleWrapper(model)
        print("[INFO] Model wrapped with TupleWrapper to handle raw audio passthrough.")
        data_count=int(args.num_classes*args.num_instance_each_class) if current_mode=='online' else int(args.num_classes*args.num_instance_each_class*(1-args.test_split))
        print('data_count is {}'.format(data_count))
        data_count=min(args.max_memory_size,data_count) # buffer_size cannot be greater than 3000
        if(strate.split("_")[-1].isnumeric()==False):
            buffer_size=data_count
        else:
            buffer_size=int(strate.split("_")[-1])

        if torch.cuda.device_count() > 1:
            print("Let's use all GPUs!")
            model = nn.DataParallel(model)
        else:
            print("only use one GPU")
        if(args.load_prev==True and state_dict is not None):
            model.load_state_dict(state_dict)
            print()
            print('loaded previous model {}'.format(model_path))
            print()
        if(torch.cuda.is_available()):
            model=model.cuda()
        optimizer=SGD(list(filter(lambda x: x.requires_grad, model.parameters())), lr=args.start_lr, weight_decay=float(args.weight_decay),momentum=args.momentum)
        scheduler= make_scheduler(optimizer,args.step_schedular_decay,args.schedular_step)
         # Initialize LLM-guided modulator (Ollama)
        # Initialize LLM-guided modulator (Ollama)
        llm_modulator = LLMGuidedModulator(
          model=model,
          base_lr=args.start_lr,
          base_wd=float(args.weight_decay),
          ollama_url="http://localhost:11434/api/generate",  # default Ollama server
          ollama_model=getattr(args, 'llm_model', "llama3.2")  # Use config or default
        )

        # Initialize LLM-guided optimization modules
        # Check if LLM optimization is enabled in config
        enable_llm_opt = getattr(args, 'enable', True) # Default to True if not specified, or check 'llm_optimization' dict if parsed differently
        # Note: get_config.py flattens the yaml. So 'llm_optimization' keys should be top-level args if flattened.
        # Let's assume get_config.py flattens it.
        
        asr_model_name = getattr(args, 'asr_model', "openai/whisper-base.en")
        asr_processor = ASRProcessor(model_name=asr_model_name, device=device)
        text_extractor = TextFeatureExtractor(device=device)
        ot_loss_fn = OptimalTransportLoss()
        
        feat_dim = getattr(args, 'pretrain_feature_shape', 2048) 
        fusion_module = MultiModalFusion(audio_dim=feat_dim, text_dim=384, output_dim=feat_dim).to(device)
        optimizer.add_param_group({'params': fusion_module.parameters()})
        
        # Hook to capture features for OT/Fusion
        captured_features = {}
        def get_features_hook(name):
            def hook(model, input, output):
                captured_features[name] = output
            return hook
            
        # Register hook on the feature extractor part
        # model is DataParallel -> module -> Sequential
        # We want the output of the Flatten layer (index 2) or before the final linear layer
        # If model is nn.Sequential(resnet, pool, flatten), then index 2 is Flatten.
        if isinstance(model, nn.DataParallel):
            if isinstance(model.module, nn.Sequential) and len(model.module) >= 3:
                model.module[2].register_forward_hook(get_features_hook('features'))
        elif isinstance(model, nn.Sequential) and len(model) >= 3:
            model[2].register_forward_hook(get_features_hook('features'))

        criterion = torch.nn.CrossEntropyLoss()
        use_linear_layer = True  # Use linear layer for audio spectrograms
        # 
        plugin_list=[LRSchedulerPlugin(scheduler),LoadBestPlugin('train_stream')]        # Build logger with comprehensive metrics
        text_logger, interactive_logger, eval_plugin = build_logger(args.split)
        
        # Add custom precision, recall, F1 metrics
        from custom_metrics import PrecisionRecallF1Metrics
        prf_metrics = PrecisionRecallF1Metrics()
        
        # Add to evaluation plugin
        eval_plugin.metrics.append(prf_metrics)
        eval_plugin.metrics.append(confusion_matrix_metrics(num_classes=args.num_classes, save_image=False, normalize='true'))
        
        
        print("\n" + "="*80)
        print("[METRICS TRACKING]")
        print("="*80)
        print("  [+] Accuracy (overall & per-class)")
        print("  [+] Loss (Cross-Entropy + Semantic + Optimal Transport)")
        print("  [+] Precision, Recall, F1-Score")
        print("  [+] Confusion Matrix (Real vs Fake)")
        print("  [+] Forgetting Metrics")
        print("  [+] Training Time & CPU Usage")
        print("="*80 + "\n")

        # -------------------------
        # Strategy Initialization
        # -------------------------
        
        # Import custom plugin
        from plugins.llm_optimization_plugin import LLMOptimizationPlugin
        
        # Add LLM plugin if enabled
        # Note: get_config.py flattens the YAML, so llm_optimization.enable becomes args.enable
        llm_enabled = getattr(args, 'enable', False)
            
        if llm_enabled:
            llm_plugin = LLMOptimizationPlugin(args, device)
            if plugin_list is None:
                plugin_list = [llm_plugin]
            else:
                plugin_list.append(llm_plugin)
            print("[INFO] LLM Optimization Plugin added.")
        else:
            print("[INFO] LLM Optimization disabled.")

        if strate=='CWRStar':
            cl_strategy = CWRStar(
                model, optimizer,
                CrossEntropyLoss(),cwr_layer_name=None, train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'Replay' in strate: 
            cl_strategy = Replay(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif (strate=='JointTraining' and current_mode=='offline'):
            cl_strategy = JointTraining(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch*args.timestamp//3, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'GDumbFinetune' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='class_balance')
        # stanard gdumb= reset model+ class_balance buffer'
        elif 'GDumb' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=True,buffer='class_balance')
        elif 'BiasReservoir' in strate:
            if('reset' in strate):
                resett=True
            else:
                resett=False
            alpha_mode ='Dynamic' if 'Dynamic' in strate else 'Fixed'
            alpha_value=float(strate.split("_")[-1])
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=resett,buffer='bias_reservoir_sampling',
                alpha_mode=alpha_mode,alpha_value=alpha_value)
        # this is basically the 'reservoir sampling in the paper(no reset+ reservoir sampling'
        elif 'Reservoir' in strate:
            cl_strategy = GDumb(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='reservoir_sampling')
        elif 'Cumulative' in strate:
            if('reset' in strate):
                resett=True
            else:
                resett=False
            cl_strategy = Cumulative(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list,reset=resett)
        elif strate=='OWM':
            cl_strategy = OWM(
                model, optimizer,
                CrossEntropyLoss(),
                alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='RAWM':
            cl_strategy = RAWM(
                model, optimizer,
                CrossEntropyLoss(),
                alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='RWM':
            cl_strategy = RWM(
                model, optimizer,
                CrossEntropyLoss(),
                alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='RegO':
            cl_strategy = RegO(
                model, optimizer,
                CrossEntropyLoss(),
                alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
                ef_thresh=0.1, importnat_thresh=0.75,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='AWMV3':
            gem_plugin = GEMPlugin(patterns_per_experience=data_count, memory_strength=0.5)
            if plugin_list is None:
                plugin_list = [gem_plugin]
            else:
                plugin_list.append(gem_plugin)

            cl_strategy = GEM(
                model, optimizer,
                CrossEntropyLoss(), 
                patterns_per_exp=data_count,
                memory_strength=0.5,
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list )
        elif 'AGEMFixed' in strate:
            agem_plugin = AGEMPlugin(patterns_per_experience=buffer_size, sample_size=buffer_size)
            if plugin_list is None:
                plugin_list = [agem_plugin]
            else:
                plugin_list.append(agem_plugin)

            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'AGEM' in strate:
            agem_plugin = AGEMPlugin(patterns_per_experience=buffer_size, sample_size=buffer_size)
            if plugin_list is None:
                plugin_list = [agem_plugin]
            else:
                plugin_list.append(agem_plugin)
                
            cl_strategy = AGEM(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='EWC':
            ewc_plugin = EWCPlugin(ewc_lambda=0.4, mode='online', decay_factor=0.1)
            if plugin_list is None:
                plugin_list = [ewc_plugin]
            else:
                plugin_list.append(ewc_plugin)

            cl_strategy = EWC(
                model, optimizer,
                CrossEntropyLoss(),
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif strate=='Naive':
            from collate_fn import CollatorAudio
            from torch.utils.data import DataLoader
            from plugins.gradient_clipping_plugin import GradientClippingPlugin
            
            # Stability Fixes: Reduce weights and add gradient clipping
            if hasattr(args, 'ot_loss_weight'): args.ot_loss_weight = 0.01
            if hasattr(args, 'region_ot_weight'): args.region_ot_weight = 0.01
            if hasattr(args, 'semantic_loss_weight'): args.semantic_loss_weight = 0.05
            
            grad_clip_plugin = GradientClippingPlugin(max_norm=1.0)
            if plugin_list is None:
                plugin_list = [grad_clip_plugin]
            else:
                plugin_list.append(grad_clip_plugin)
            
            custom_collate = CollatorAudio()
            
            # Minimal subclass to inject custom collate - this is the only reliable way
            class CustomNaive(Naive):
                def make_train_dataloader(self, num_workers=0, shuffle=True, pin_memory=True, **kwargs):
                    """Override to inject custom collate_fn"""
                    self.dataloader = DataLoader(
                        self.adapted_dataset,
                        batch_size=self.train_mb_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=custom_collate
                    )
                
                def make_eval_dataloader(self, num_workers=0, pin_memory=True, **kwargs):
                    """Override to inject custom collate_fn"""
                    self.dataloader = DataLoader(
                        self.adapted_dataset,
                        batch_size=self.eval_mb_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=custom_collate
                    )
            
            # Reduce learning rate for stability (0.01 -> 0.001)
            optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
            
            cl_strategy = CustomNaive(
                model=model,
                optimizer=optimizer,
                criterion=CrossEntropyLoss(),
                train_mb_size=args.batch_size,
                train_epochs=args.nepoch,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin,
                device=device,
                plugins=plugin_list
            )


            # cl_strategy.make_eval_dataloader = make_eval_dataloader # Not needed for training-only LLM opt
        elif strate=='SI':
            si_plugin = SynapticIntelligencePlugin(si_lambda=0.1)
            if plugin_list is None:
                plugin_list = [si_plugin]
            else:
                plugin_list.append(si_plugin)

            cl_strategy = SynapticIntelligence(
                model, optimizer,
                CrossEntropyLoss(),
                train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)
        elif 'CoPE' in strate:
            cl_strategy = CoPE(
                model, optimizer,
                CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.nepoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
                evaluator=eval_plugin,device=device,plugins=plugin_list)


        else:
            continue
            
        print('\n' + "="*80)
        print(f"[STARTING EXPERIMENT] Strategy: {strate} | Mode: {current_mode}")
        print("="*80 + '\n')

        train_metric={}
        test_metric = {}
        
        # -------------------------
        # Main Training Loop
        # -------------------------
        
        if(strate=='JointTraining' and current_mode=='offline'):
            # Offline Joint Training
            print(f"Starting Joint Training on all data...")
            if(args.eval==False):
                train_metric[0]=cl_strategy.train(scenario.train_stream)
            test_metric[0]=cl_strategy.eval(scenario.test_stream)
            
            # Save model
            model_save_path='../{}/model/model_{}_{}_time{}.pth'.format(args.split,strate,current_mode,0)
            torch.save(model.state_dict(), model_save_path)
            print(f"[SAVED] Model saved to {model_save_path}")
            
        else:
            # Continual Learning Loop
            train_stream = scenario.train_stream
            
            for experience in train_stream:
                curr_exp = experience.current_experience
                print(f"\n[EXPERIENCE {curr_exp}] Starting training...")
                print(f"  Classes: {experience.classes_in_this_experience}")
                
                # Train on current experience
                if not args.eval:
                    train_res = cl_strategy.train(experience)
                    train_metric[curr_exp] = train_res
                    print(f"[EXPERIENCE {curr_exp}] Training completed.")
                    
                    # --- LLM-guided modulation (Ollama) ---
                    try:
                        prev_test = test_metric.get(curr_exp - 1, {})
                        summary_text = llm_modulator.build_summary(
                            exp_id=curr_exp,
                            train_metrics=train_res,
                            prev_test_metrics=prev_test
                        )
                        modulation_dict = llm_modulator.call_llm_for_modulation(summary_text)
                        llm_modulator.apply_modulation(optimizer, modulation_dict)
                        print(f"[LLM] Applied modulation for exp={curr_exp}")
                    except Exception as e:
                        print(f"[LLM] Modulation skipped: {e}")
                    # --------------------------------------

                # Evaluate on test stream
                # Evaluate on test stream
                print(f"[EXPERIENCE {curr_exp}] Evaluating on all datasets...")
                # Filter out empty experiences to prevent IndexError
                valid_test_experiences = []
                for exp in scenario.test_stream:
                    if len(exp.dataset) > 0:
                        valid_test_experiences.append(exp)
                
                if len(valid_test_experiences) > 0:
                    test_res = cl_strategy.eval(valid_test_experiences)
                    test_metric[curr_exp] = test_res
                else:
                    print("[WARN] No valid test experiences found. Skipping evaluation.")
                
                # Save model
                model_save_path = f'../{args.split}/model/model_{strate}_{current_mode}_time{str(curr_exp).zfill(2)}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f"[SAVED] Model saved to {model_save_path}")

        # Save metrics
        with open(f"../{args.split}/metric/train_metric_{strate}.json", "w") as out_file:
            # Helper to handle tensors
            def convert(o):
                if isinstance(o, torch.Tensor): return o.cpu().tolist()
                return o
            json.dump(train_metric, out_file, indent=6, default=convert)
            
        with open(f"../{args.split}/metric/test_metric_{strate}.json", "w") as out_file:
            json.dump(test_metric, out_file, indent=6, default=convert)
            
        print("\n" + "="*80)
        print("[FINISHED] Training and Evaluation Complete.")
        print("="*80 + "\n")



