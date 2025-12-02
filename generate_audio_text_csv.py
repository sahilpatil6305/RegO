"""
Script to generate CSV file with audio file paths, dataset info, and ASR transcriptions.
This verifies that the audio-to-text feature is working correctly.
"""

import torch
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add RegO modules to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_guided_optimization import ASRProcessor
from load_dataset import CLEARDataset
import yaml

def generate_audio_text_csv(args):
    """
    Generate CSV with audio paths, dataset info, and transcriptions.
    
    Columns:
    - audio_path: Path to audio file
    - dataset: Dataset name (e.g., 'clear10')
    - class_label: Class index
    - task_id: Task/experience ID
    - transcription: ASR-generated text
    """
    
    print("="*80)
    print("Audio-to-Text CSV Generator")
    print("="*80)
    
    # Load configuration
    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ASR processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    asr_model = config.get('asr_model', 'openai/whisper-base.en')
    print(f"ASR Model: {asr_model}")
    
    asr_processor = ASRProcessor(model_name=asr_model, device=device)
    print("✓ ASR Processor initialized")
    
    # Load dataset
    print(f"\nLoading dataset from: {config['data_root']}")
    dataset = CLEARDataset(
        root=config['data_root'],
        train=True,
        download=False
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Process samples
    results = []
    num_samples = min(args.num_samples, len(dataset))
    
    print(f"\nProcessing {num_samples} samples...")
    for idx in tqdm(range(num_samples)):
        try:
            # Get sample
            sample = dataset[idx]
            
            # Unpack based on dataset return format
            if isinstance(sample[0], tuple):
                # ((spec, raw), label, task_id)
                spec, raw_audio = sample[0]
                label = sample[1]
                task_id = sample[2]
            else:
                # Fallback
                raw_audio = sample[0]
                label = sample[1]
                task_id = sample[2] if len(sample) > 2 else 0
            
            # Ensure raw_audio is on correct device and has batch dimension
            if raw_audio.dim() == 1:
                raw_audio = raw_audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            elif raw_audio.dim() == 2:
                raw_audio = raw_audio.unsqueeze(0)  # (1, C, T)
            
            raw_audio = raw_audio.to(device)
            
            # Generate transcription
            transcripts = asr_processor.transcribe(raw_audio)
            transcription = transcripts[0] if isinstance(transcripts, list) else str(transcripts)
            
            # Get audio path (if available)
            audio_path = f"sample_{idx}"
            if hasattr(dataset, 'data') and idx < len(dataset.data):
                audio_path = str(dataset.data[idx])
            
            # Store result
            results.append({
                'sample_id': idx,
                'audio_path': audio_path,
                'dataset': 'clear10',
                'class_label': int(label),
                'task_id': int(task_id),
                'transcription': transcription,
                'audio_length': raw_audio.shape[-1]
            })
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = args.output
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ CSV saved to: {output_file}")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"\nFirst 5 transcriptions:")
    print("-"*80)
    for idx, row in df.head().iterrows():
        print(f"[{row['sample_id']}] Class {row['class_label']}: {row['transcription'][:100]}...")
    print("="*80)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio-to-text CSV")
    parser.add_argument('--yaml', type=str, default='yaml/clear10/train.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process')
    parser.add_argument('--output', type=str, default='audio_text_transcriptions.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    df = generate_audio_text_csv(args)
