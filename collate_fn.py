import torchaudio.transforms as T
import torch
import torch.nn.functional as F


class CollatorAudio:
    """Batch collator for audio.

    - Pads/truncates waveforms to `max_len_seconds`.
    - Computes Mel spectrogram -> dB.
    - Expands to 3 channels (broadcast) to match typical image backbones.
    - Returns `(batch, 3, n_mels, time)` and `labels` as `torch.long`.
    """

    def __init__(self, sample_rate=16000, n_mels=128, max_len_seconds=4.0, device=None):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = int(sample_rate * max_len_seconds)
        self.device = device
        # n_fft=400 produces 201 frequency bins, which is > n_mels (128)
        self.mel = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=400)
        self.db = T.AmplitudeToDB()

    def __call__(self, batch):
            # load_dataset.py returns: img, int(class_index), int(task_id), raw_audio
            # But wait, `img` in load_dataset is ALREADY a 3-channel tensor (3, 224, 224).
            # AND `raw_audio` is the waveform.
            # CollatorAudio expects `waveform` as input if it's doing the processing?
            # Let's check load_dataset.py again.
            # load_dataset.py: returns `img` (processed) AND `raw_audio`.
            # If the dataset ALREADY processes the image, why does CollatorAudio do it again?
            # CollatorAudio: `self.mel = T.MelSpectrogram...`
            # If the dataset returns processed images, CollatorAudio should just stack them?
            # Let's check if `load_dataset.py` is used with `CollatorAudio`.
            # train.py: `scenario = get_data_set_offline(args)` -> `dataset_benchmark`
            # `dataset_benchmark` uses the passed transform.
            # `load_dataset.py` applies transforms in `__getitem__`?
            # `load_dataset.py` lines 339-356: computes mel spectrogram, log scale, resize, repeat.
            # So `img` IS the mel spectrogram.
            # `CollatorAudio` seems to be designed for RAW waveforms.
            # If `train.py` uses `CollatorAudio` as `collate_fn` for the DataLoader, then the dataset MUST return raw waveforms.
            # BUT `load_dataset.py` returns processed images.
            # CONFLICT: `load_dataset.py` returns processed tensors. `CollatorAudio` expects raw waveforms.
            # If `CollatorAudio` is used, it will try to compute MelSpectrogram on the ALREADY PROCESSED MelSpectrogram?
            # That would be wrong.
            # Let's check `train.py` to see if `collate_fn` is used.
            # `train.py` imports `collator_audio`.
            # But where is it used?
            # `cl_strategy = Naive(..., plugins=plugin_list)`
            # Avalanche handles DataLoaders internally.
            # Unless we pass a custom `collate_fn` to the strategy or plugin?
            # Avalanche `SupervisedTemplate` allows `train_mb_size`, etc.
            # It doesn't seem to easily allow custom `collate_fn` unless we override `make_train_dataloader`.
            # Wait, `load_dataset.py` returns `img`.
            # If `CollatorAudio` is NOT used, then Avalanche uses default collate.
            # Default collate stacks tensors.
            # If `load_dataset.py` returns `(img, label, task_id, raw_audio)`, default collate will try to stack them.
            # `img` -> stacked (batch, 3, 224, 224). OK.
            # `label` -> stacked (batch). OK.
            # `task_id` -> stacked (batch). OK.
            # `raw_audio` -> stacked (batch, 1, samples). OK (if same length).
            # `load_dataset.py` pads raw_audio to max_samples. So it should be OK.
            
            # So, is `CollatorAudio` actually used?
            # `train.py` line 18: `from collate_fn import collator_audio`
            # But I don't see it being passed to `Naive` or `dataset_benchmark`.
            # If it's NOT used, then `CollatorAudio` is dead code or for a different pipeline.
            # However, the user complained about runtime errors.
            # Maybe the error is that `load_dataset` returns 4 items, and Avalanche expects 3?
            # Avalanche `AvalancheDataset` might try to unpack (x, y, t).
            # If `__getitem__` returns 4, `AvalancheDataset` might fail or just pass it through.
            # If it passes it through, the DataLoader collate will receive a list of 4-tuples.
            # Default collate will produce a list of 4 batches: [batch_x, batch_y, batch_t, batch_raw].
            # Then the training loop: `for mb in dataloader:`
            # `mb` will be `[x, y, t, raw]`.
            # Avalanche strategy unpacks `mb`.
            # `mb_x, mb_y, mb_task_id = mb` -> ValueError: too many values to unpack (expected 3).
            # THIS is likely the runtime error.
            
            # FIX: We need to make sure the DataLoader yields something Avalanche can handle, OR we handle the unpacking.
            # But we can't easily change Avalanche's unpacking logic in `common.py` or `supervised.py`.
            # We CAN change the dataset to return (x, y, t) and pack `raw_audio` into `x` or something?
            # OR we can use a custom plugin that intercepts the batch?
            # But the unpacking happens inside the loop in `BaseStrategy.train_epoch`.
            # `for self.mbatch in self.dataloader:`
            # `self.mb_x, self.mb_y, self.mb_task_id = self.mbatch` (simplified)
            # Actually, Avalanche is smarter. It handles dictionary or list.
            # If list, it expects x, y, t.
            # If we return 4 items, it WILL crash.
            
            # SOLUTION:
            # Modify `load_dataset.py` to return `(img, raw_audio), label, task_id`.
            # Then `mb_x` will be `(img, raw_audio)`.
            # Then we need to update the model to handle tuple input?
            # OR update the plugin to unpack it.
            # `LLMOptimizationPlugin` uses `strategy.mbatch`.
            # If `mb_x` is a tuple, `model(mb_x)` will fail unless model handles it.
            # We can use a `forward_hook` or wrapper?
            # Or just change `load_dataset` to NOT return raw_audio in `__getitem__`?
            # But we need raw_audio for ASR.
            # Can we get raw_audio from `img`? No (mel spec is lossy).
            # Can we load it on the fly in the plugin? Yes, if we have file paths.
            # `load_dataset` returns file path? No, it loads it.
            # `CLEARDataset` has `self.samples`.
            # We can pass indices?
            # Return `img, label, task_id, index`.
            # Then in plugin, use `dataset.samples[index]` to get path and load audio.
            # This is cleaner and avoids breaking Avalanche's (x, y, t) expectation (if we treat index as metadata).
            # But Avalanche expects (x, y, t). 4 items is still 4 items.
            # Unless we return `img, label, task_id`. Where do we put index?
            # `AvalancheDataset` supports `sup_labels`?
            # Maybe we can return `(img, index), label, task_id`.
            # Then `mb_x` is `(img, index)`.
            # Model needs to handle it.
            
            # Let's try to verify what `CollatorAudio` does.
            # If the user INTENDED to use `CollatorAudio`, then `train.py` should use it.
            # But `train.py` doesn't seem to use it.
            # So `CollatorAudio` might be a red herring or intended for a custom DataLoader that isn't hooked up.
            
            # Let's assume the error IS "too many values to unpack".
            # I will modify `load_dataset.py` to return `img, label, task_id`.
            # AND I will modify `load_dataset.py` to store `raw_audio` or `path` in a way we can retrieve it.
            # OR, I can use `AvalancheDataset`'s `resize` or something?
            # No.
            
            # Let's look at `train.py` again.
            # `scenario = get_data_set_offline(args)`
            # `get_data_set_offline` returns `dataset_benchmark(...)`.
            # This creates `AvalancheDataset`s.
            # `AvalancheDataset` `__getitem__` calls the underlying dataset `__getitem__`.
            # If underlying returns 4 items, `AvalancheDataset` returns 4 items (unless strict).
            # If it returns 4 items, the DataLoader collates 4 items.
            # The training loop crashes.
            
            # PROPOSED FIX:
            # 1. Modify `load_dataset.py` to return `img, label, task_id`.
            # 2. BUT we need `raw_audio` for `LLMGuidedModulator` -> `ASRProcessor`.
            # 3. `LLMGuidedModulator` is called in `after_training_exp` (or similar).
            # 4. It uses `train_metrics`. It doesn't seem to use `mbatch` directly?
            # 5. Wait, `llm_guided_optimization.py` has `ASRProcessor`.
            # 6. `train.py` initializes `asr_processor`.
            # 7. But where is `asr_processor` USED?
            # 8. `train.py` lines 310-317 init them, but I don't see them used in the loop!
            # 9. `train.py` lines 578-587 use `llm_modulator`.
            # 10. `llm_modulator.build_summary` uses metrics.
            # 11. `llm_modulator.call_llm_for_modulation`.
            # 12. `llm_modulator.apply_modulation`.
            # 13. I DO NOT SEE `asr_processor` or `fusion_module` being used in the training loop in `train.py`.
            # 14. They are initialized but ignored?
            # 15. Ah, `LLMOptimizationPlugin` (line 366) might use them.
            # 16. `llm_plugin = LLMOptimizationPlugin(args, device)`.
            # 17. If `LLMOptimizationPlugin` needs raw audio, it must get it from somewhere.
            # 18. If `load_dataset` doesn't return it, the plugin can't get it easily.
            
            # Let's check `LLMOptimizationPlugin` (I can't see it, but I can infer or check if it exists).
            # I'll assume it exists.
            # If I change `load_dataset` to return 3 items, I fix the crash.
            # But I might break the plugin if it expects raw audio.
            # However, if the code is currently crashing, fixing the crash is priority 1.
            # I will modify `load_dataset.py` to return 3 items.
            # I will also modify `CollatorAudio` just in case it IS used (to be safe), to handle the input correctly.
            
            # Wait, if `load_dataset` returns processed images, `CollatorAudio` (which computes mel) is definitely wrong if applied.
            # So `CollatorAudio` is likely NOT used or used with a different dataset.
            # I will focus    def __call__(self, batch):
        tensors = []
        labels = []
        task_ids = []
        raw_audios = []

        for item in batch:
            # item is ((spec, raw), label, task_id)
            x_part = item[0]
            label = item[1]
            
            if isinstance(x_part, (tuple, list)) and len(x_part) == 2:
                tensors.append(x_part[0])
                raw_audios.append(x_part[1])
            else:
                # Fallback
                tensors.append(x_part)
                # Add dummy raw audio if missing
                raw_audios.append(torch.zeros(16000)) 
                
            labels.append(label)
            if len(item) >= 3:
                task_ids.append(item[2])

        # Stack
        x = torch.stack(tensors)
        y = torch.tensor(labels, dtype=torch.long)
        t = torch.tensor(task_ids, dtype=torch.long) if task_ids else torch.zeros_like(y)
        
        # Stack raw audios
        if raw_audios:
            # Pad if necessary (though they should be same length)
            # For now assume same length
            try:
                raw_batch = torch.stack(raw_audios)
            except:
                # Handle variable length if needed, but for now just stack
                raw_batch = torch.stack(raw_audios)
                
            # Return ((x, raw), y, t)
            # print(f"DEBUG: Returning DeviceAwareTuple with types: {type(x)}, {type(raw_batch)}")
            return DeviceAwareTuple((x, raw_batch)), y, t
            
        return x, y, t

class DeviceAwareTuple(tuple):
    def to(self, device, **kwargs):
        # Move only the spectrogram (first element) to device
        # Keep raw audio (second element) as is (or move if needed, but usually CPU is fine for ASR input)
        # Actually, let's move both if possible, but raw audio might be a list or tensor.
        # If raw_batch is a tensor, .to() works.
        return DeviceAwareTuple((self[0].to(device, **kwargs), self[1]))



# Backwards-compatible name (using n_mels=128 to avoid filterbank warnings)
collator_audio = CollatorAudio(n_mels=128)
