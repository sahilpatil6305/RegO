from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip,Resize
from torch.utils.data import Dataset,Subset
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import NCScenario, nc_benchmark,dataset_benchmark,ni_benchmark
from PIL import Image
import torch
from avalanche.benchmarks  import benchmark_with_validation_stream
from parse_data_path import *
import torch
import torchaudio
import torchaudio.transforms as T
from collate_fn import CollatorAudio
import torch
import torchaudio
import torchaudio.transforms as T
'''
timestamp_index stand for, for each timestamp, the index of instance in the txt file, since each subset represent
one timestamp data, thus we need to have the index of data of each timestamp
timestamp_index[0] == the set of index of data belong to bucket 1
'''
def get_instance_time(args,idx,all_timestamp_index):
    for index,list in enumerate(all_timestamp_index):
        if(idx in list):
            return index
    assert False, "couldn't find timestamp info for data with index {}".format(idx)

def get_feature_extract_loader(args):
    dataset=CLEARDataset(args,data_txt_path='../{}/data_cache/data_all_path.txt'.format(args.split),stage='all')
    all_timestamp_index=dataset.get_timestamp_index()
    return dataset,all_timestamp_index


class CLEARDataset(Dataset):
    def __init__(self, args,data_txt_path,stage):
        assert stage in ['train','test','all']
        print('Preparing {}'.format(stage))
        self.args=args
        self.n_classes=args.num_classes
        self.n_experiences=args.timestamp
        self.stage=stage
        if(os.path.isfile(data_txt_path)==False):
            print('loading data_list from folder')
            parse_data_path(args)
        else:
            print('loaded exist data_list')
        # data_txt_path: '../temp_folder/data_cache/data_all_path.txt'
        self.prepare_data(data_txt_path)
        # Ensure targets are long tensors for cross-entropy compatibility
        import torch as _torch
        self.targets = _torch.tensor(self.targets, dtype=_torch.long)
        print('Using split {}'.format(self.args.split))
        # self.train_transform,self.test_transform=self.get_transforms()
        # Pre-create audio transforms to avoid re-creating them per-sample (expensive)
        try:
            self.target_sr = 16000
            self.n_mels = 224
            self._mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.target_sr, n_mels=self.n_mels)
        except Exception:
            # If torchaudio not available at init time, create lazily in __getitem__
            self._mel_transform = None
        
        # Buffer to store raw audio for current batch (accessed by plugin)
        self.raw_audio_buffer = {}
    def get_timestamp_index(self):
        return self.timestamp_index
    # def prepare_data(self,data_txt_path):
    #     # save_path='../{}/data_cache/{}_save'.format(self.args.split,self.stage)
    #     # if(os.path.isfile(save_path+'.npy')):
    #     #     self.targets,self.samples,self.timestamp_index=np.load(save_path+'.npy',allow_pickle=True)
    #     # else:
    #     samples=[]
    #     targets=[]
    #     timestamp_index=[[] for i in range(self.n_experiences)]
    #     index=0
    #     with open(data_txt_path,'r') as file:
    #         title=file.readline()
    #         while (True):
    #             line=file.readline()
    #             if(line==''):
    #                 break
    #             #'/data3/zhiqiul/yfcc_dynamic_10/dynamic_300/images/bucket_6/racing/6111026970.jpg 8 6\n'
    #             # line_list=line.split()
    #             # targets.append(int(line_list[1]))
    #             # timestamp_index[int(line_list[2])-1].append(index)
    #             # samples.append(line_list)

    #             # Split from the right side: [filepath, class_index, timestamp]
    #             parts = line.strip().rsplit(" ", 2)
    #             if len(parts) != 3:
    #                 raise ValueError(f"Invalid line format: {line}")

    #             filepath, class_index, timestamp = parts
    #             targets.append(int(class_index))
    #             timestamp_index[int(timestamp)-1].append(index)
    #             samples.append([filepath, class_index, timestamp])
    #             index=index+1
    #             if(index%10000==0):
    #                 print('finished processing data {}'.format(index))
                
    #     self.targets=targets
    #     self.samples=samples
    #     self.timestamp_index=timestamp_index
    # def prepare_data(self, data_txt_path):
    #         samples = []
    #         targets = []
    #         timestamp_index = [[] for _ in range(self.n_experiences)]
    #         index = 0

    #         with open(data_txt_path, 'r') as file:
    #             title = file.readline()  # skip header if exists
    #             while True:
    #                 line = file.readline()
    #                 if line == '':
    #                     break

    #                 parts = line.strip().split()  # split by whitespace

    #                 if len(parts) < 2:
    #                     raise ValueError(f"Invalid line format: {line}")

    #                 filepath = parts[0]

    #                 if len(parts) == 2:
    #                     # only filepath and class_index
    #                     class_index = parts[1]
    #                     timestamp = None
    #                     samples.append([filepath, class_index])
    #                 else:
    #                     # filepath, class_index, timestamp
    #                     class_index = parts[1]
    #                     timestamp = parts[2]
    #                     timestamp_index[int(timestamp) - 1].append(index)
    #                     samples.append([filepath, class_index, timestamp])

    #                 targets.append(int(class_index))
    #                 index += 1

    #                 if index % 10000 == 0:
    #                     print('finished processing data {}'.format(index))

    #         self.targets = targets
    #         self.samples = samples
    #         self.timestamp_index = timestamp_index
    def prepare_data(self, data_txt_path):
        """
        Robust parsing:
        - Expect tab-separated: filepath \t class_index [\t optional_label]
        - If line malformed, try fallback repairs:
            1) split by whitespace and assume last token is class_index
            2) if class_index cannot be parsed, skip the line and log it
        - Do NOT raise on bad lines; collect skipped lines in a log (printed + saved).
        """
        import os
        self.samples = []
        self.targets = []
        self.timestamp_index = [[] for _ in range(self.n_experiences)]
        index = 0
        skipped = []

        with open(data_txt_path, 'r') as file:
            header = file.readline()  # skip header if present
            for lineno, line in enumerate(file, start=2):  # start=2 because header line is 1
                raw_line = line.rstrip("\n")
                line = raw_line.strip()
                if line == '':
                    continue

                filepath = None
                class_index = None

                # Prefer tab-separated parsing (safe when file is properly tab-separated)
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split()  # fallback to whitespace split

                # Case A: normal expected case: at least 2 parts
                if len(parts) >= 2:
                    # Use first two columns as filepath and class_index
                    filepath_candidate = parts[0]
                    class_candidate = parts[1]
                    # If there are more parts, ignore extras (or you can keep parts[2] as label)
                    try:
                        class_index = int(class_candidate)
                        filepath = filepath_candidate
                    except Exception:
                        # try alternative: last token as class_index, rest joined as filepath
                        try:
                            class_index = int(parts[-1])
                            filepath = ' '.join(parts[:-1])
                        except Exception:
                            filepath = None
                            class_index = None
                else:
                    # len(parts) < 2: try a repair: assume last token is class_index
                    parts_ws = line.split()
                    if len(parts_ws) >= 2:
                        try:
                            class_index = int(parts_ws[-1])
                            filepath = ' '.join(parts_ws[:-1])
                        except Exception:
                            filepath = None
                            class_index = None
                    else:
                        filepath = None
                        class_index = None

                # Final validation: filepath exists and class_index is valid int
                if filepath is None or class_index is None:
                    skipped.append((lineno, raw_line))
                    # print brief info and continue
                    print(f"[prepare_data] Skipping malformed line {lineno}: {raw_line}")
                    continue

                # Optionally: check if filepath exists on disk; if not, also skip (or keep as is)
                if not os.path.isfile(filepath):
                    # If you prefer to treat missing files as skip, uncomment:
                    print(f"[prepare_data] Warning: file not found for line {lineno}, skipping: {filepath}")
                    skipped.append((lineno, raw_line))
                    continue

                # All good: append sample and target
                self.samples.append([filepath, class_index])
                self.targets.append(class_index)
                # Append to timestamp index 0 (or decide based on your timestamp logic)
                self.timestamp_index[0].append(index)

                index += 1
                if index % 10000 == 0:
                    print(f'Finished processing data {index}')

        # Save/print summary of skipped lines
        if skipped:
            print(f"[prepare_data] Finished with {len(skipped)} skipped lines. See 'skipped_lines.log' for details.")
            try:
                log_path = os.path.join(os.path.dirname(data_txt_path), "skipped_lines.log")
                with open(log_path, "w") as lf:
                    for lineno, raw_line in skipped:
                        lf.write(f"{lineno}\t{raw_line}\n")
                print(f"[prepare_data] Skipped lines written to: {log_path}")
            except Exception as e:
                print(f"[prepare_data] Could not write skipped_lines.log: {e}")
        else:
            print("[prepare_data] All lines parsed successfully. No skipped lines.")

        # assign timestamp_index ensured above
        self.timestamp_index = self.timestamp_index


            # save=(self.targets,self.samples,self.timestamp_index)
            # np.save(save_path,save)

    def __len__(self):
        return len(self.samples)
    
    # def __getitem__(self,index):
    #     import os
    #     os.makedirs('../{}/buffered_data/train'.format(self.args.split),exist_ok=True)
    #     os.makedirs('../{}/buffered_data/test'.format(self.args.split),exist_ok=True)
    #     os.makedirs('../{}/buffered_data/all'.format(self.args.split),exist_ok=True)
    #     file_path='../{}/buffered_data/{}/{}.npy'.format(self.args.split,self.stage,str(index))
    #     # if(os.path.isfile(file_path)):
    #     #     print('loaded data')
    #     #     image_array,label= np.load(file_path,allow_pickle=True)
    #     #     sample=Image.fromarray(image_array)
    #     #     return sample,label
    #     # else:
    #     '''
    #     when using pre-train feature and data_folder_path had already be updated
    #     When generating pretrain feature, the data_folder_path is original image path
    #     When finish generating pretrain feature, the data_folder_path is feature path
    #     '''
    #     if(self.args.pretrain_feature!='None' and self.args.data_folder_path.endswith('feature')==True):
    #         sample, label = torch.load(self.samples[index][0])[0],self.samples[index][1]
    #     else:
    #         sample, label = Image.open(self.samples[index][0]),self.samples[index][1]
    #         array=np.array(sample)
    #         # some image may have 4 channel (alpha)
    #         if(array.shape[-1]==4):
    #             array=array[:,:,:3]
    #         elif(array.shape[-1]==1):
    #             array=np.concatenate((array, array, array), axis=-1)
    #         elif(len(array.shape)==2):
    #             array=np.stack([array,array,array],axis=-1)
    #         # import pdb;pdb.set_trace()
    #         # array=np.ones(array.shape,dtype='uint8')*int(get_instance_time(self.args,index,self.timestamp_index)) # for debug
    #         sample=Image.fromarray(array)
    #     # result= array,label
    #     # np.save('./buffered_data/{}/{}'.format(self.stage,str(index)),result)
    #     return sample,label

    import torchaudio
    import torch
    import torch.nn.functional as F

    def __getitem__(self, index):
        # load wav (use torchaudio when available, otherwise fallback to soundfile)
        wav_path, class_index = self.samples[index]
        
        # Target configuration
        target_sr = 16000
        max_duration = 4.0
        max_samples = int(target_sr * max_duration)
        
        try:
            waveform, sr = torchaudio.load(wav_path)  # returns (channels, samples)
        except Exception:
            # Fallback to soundfile (pip install soundfile) which works reliably
            try:
                import soundfile as sf
                data, sr = sf.read(wav_path, dtype='float32')
                import numpy as _np
                if data.ndim == 1:
                    waveform = torch.from_numpy(_np.expand_dims(data, 0))
                else:
                    # soundfile returns (samples, channels)
                    waveform = torch.from_numpy(data.T)
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                # Return dummy data to avoid crashing
                waveform = torch.zeros(1, max_samples)
                sr = target_sr

        # convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

        # Pad or crop to fixed length for batching
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            pad_len = max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))
            
        # Keep raw waveform for ASR (1, 64000)
        raw_audio = waveform.clone()

        # 1) compute mel spectrogram (use pre-created transform when available)
        if getattr(self, '_mel_transform', None) is None:
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_mels=224)(waveform)
        else:
            mel_spec = self._mel_transform(waveform)

        # 2) log scale
        log_mel = torch.log(mel_spec + 1e-9)  # (1, n_mels, time)

        # 3) normalize per-sample (optional)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        # 4) resize to fixed time dimension 224 -> final (1,224,224)
        log_mel = F.interpolate(log_mel.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        log_mel = log_mel.squeeze(0)  # (1,224,224)

        # 5) convert to 3 channels for ResNet: (3,224,224)
        spectrogram = log_mel.repeat(3, 1, 1)

        # task id: Avalanche needs task label (use 0 if single-task)
        task_id = 0

        # Return (spectrogram, raw_audio) tuple
        # This allows standard collate handling (with custom collate) and plugin access via mbatch
        return (spectrogram, raw_audio), int(class_index), int(task_id)



import os
import torch
import torchaudio
from torch.utils.data import Dataset

audio_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ensure fixed 224x224 for image-based models
    transforms.Normalize(mean=[0.5], std=[0.5])  # optional normalization
])

class AudioDataset(Dataset):
    def __init__(self, samples, args, sample_rate=16000, max_duration=5.0):
        """
        samples: list of tuples [(wav_path, class_index), ...]
        args: argument object (for split info, etc.)
        """
        self.samples = samples
        self.args = args
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_duration)   # 🔑 define max_len here

        # Optional: create buffered folders (if needed for caching)
        os.makedirs(f'../{self.args.split}/buffered_data/train', exist_ok=True)
        os.makedirs(f'../{self.args.split}/buffered_data/test', exist_ok=True)
        os.makedirs(f'../{self.args.split}/buffered_data/all', exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        wav_path, class_index = self.samples[index]

        # Load waveform
        waveform, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or crop to fixed length
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_len = self.max_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))

        # ✅ Convert to log-mel spectrogram
        mel = audio_to_log_mel(waveform, sample_rate=self.sample_rate)  
        # should return tensor like (1, H, W)

        # ✅ If spectrogram is single-channel, expand to 3-channel (for ResNet etc.)
        if mel.shape[0] == 1:
            mel = mel.repeat(3, 1, 1)

        # ✅ Apply resize/normalize transform
        mel = audio_transform(mel)

        return mel, int(class_index)

# Separate file or same file: collator_audio
def audio_to_log_mel(waveform, sample_rate=16000, n_mels=128, target_size=(224, 224)):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels
    )(waveform)

    log_mel = torch.log(mel_spec + 1e-9)  # (1, n_mels, time)

    # resize to target_size
    log_mel = F.interpolate(log_mel.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False)
    log_mel = log_mel.squeeze(0)  # (1, 224, 224)

    # repeat to 3 channels
    log_mel = log_mel.repeat(3, 1, 1)  # (3, 224, 224)

    return log_mel



def collator_audio(batch):
    tensors = []
    labels = []
    for waveform, label in batch:
        mel_spec = T.MelSpectrogram(sample_rate=16000, n_mels=224)(waveform)
        log_mel_spec = torch.log1p(mel_spec)
        tensor = log_mel_spec.repeat(3,1,1)
        tensors.append(tensor)
        labels.append(label)
    return torch.stack(tensors), torch.tensor(labels)



class CLEARSubset(Dataset):
    def __init__(self, dataset, indices, targets,bucket):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.indices=indices
        self.targets = targets.numpy() # need to be in numpy(thus set of targets have only 10 elem,rather than many with tensor)
        self.bucket=bucket
    def get_indice(self):
        return self.indices
    def get_bucket(self):
        return self.bucket
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        assert int(self.dataset[idx][1])==target
        return (image, target)
    def __len__(self):
        return len(self.targets)
def get_transforms(args):
    # Note that this is not exactly imagenet transform/moco transform for val set
    # Because we resize to 224 instead of 256
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if(args.pretrain_feature!='None'):
        train_transform,test_transform=None,None
    return train_transform, test_transform
def get_data_set_offline(args):
    train_Dataset=CLEARDataset(args,data_txt_path='../{}/data_cache/data_train_path.txt'.format(args.split),stage='train')
    print("Number of train data is {}".format(len(train_Dataset)))
    test_Dataset=CLEARDataset(args,data_txt_path='../{}/data_cache/data_test_path.txt'.format(args.split),stage='test')
    print("Number of test data is {}".format(len(test_Dataset)))
    # import pdb;pdb.set_trace()
    n_experiences=args.timestamp
    train_timestamp_index,test_timestamp_index=train_Dataset.get_timestamp_index(),test_Dataset.get_timestamp_index()
    train_transform,test_transform=get_transforms(args)
    # print(train_timestamp_index)

    list_train_dataset = []
    list_test_dataset = []
    for i in range(n_experiences):
        # choose a random permutation of the pixels in the image
        bucket_index=train_timestamp_index[i]
        train_sub = CLEARSubset(train_Dataset,bucket_index,train_Dataset.targets[bucket_index],i)
        train_set = train_sub

        bucket_index=test_timestamp_index[i]
        test_sub = CLEARSubset(test_Dataset,bucket_index,test_Dataset.targets[bucket_index],i)
        test_set = test_sub

        list_train_dataset.append(train_set)
        list_test_dataset.append(test_set)
    # Custom collate function will be passed to strategy instead
    return dataset_benchmark(
        list_train_dataset, 
        list_test_dataset, 
        train_transform=train_transform,
        eval_transform=test_transform
    )
    
    
    # return ni_benchmark(
    #     list_train_dataset, 
    #     list_test_dataset, 
    #     n_experiences=len(list_train_dataset), 
    #     shuffle=False, 
    #     balance_experiences=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)

    # return nc_benchmark(
    #     list_train_dataset,
    #     list_test_dataset,
    #     n_experiences=len(list_train_dataset),
    #     task_labels=True,
    #     shuffle=False,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)

    # return nc_benchmark(
        # list_train_dataset,
        # list_test_dataset,
        # n_experiences=len(list_train_dataset),
        # task_labels=True,
        # shuffle=False,
        # class_ids_from_zero_in_each_exp=True,
        # one_dataset_per_exp=True,
        # train_transform=train_transform,
        # eval_transform=test_transform,
        # seed=args.random_seed)
    # valid_benchmark = benchmark_with_validation_stream(
    #         initial_benchmark_instance, 20, shuffle=False)
    # return valid_benchmark


def get_data_set_online(args):
    all_Dataset=CLEARDataset(args,data_txt_path='../{}/data_cache/data_all_path.txt'.format(args.split),stage='all')
    print("Number of all data is {}".format(len(all_Dataset)))
    n_experiences=args.timestamp
    all_timestamp_index=all_Dataset.get_timestamp_index()
    train_transform,test_transform=get_transforms(args)

    list_all_dataset = []

    for i in range(n_experiences):
        # choose a random permutation of the pixels in the image
        bucket_index=all_timestamp_index[i]
        all_sub = CLEARSubset(all_Dataset,bucket_index,all_Dataset.targets[bucket_index],i)
        all_set = all_sub
        list_all_dataset.append(all_set)
    return dataset_benchmark(
        list_all_dataset, 
        list_all_dataset, 
        train_transform=train_transform,
        eval_transform=test_transform)


    # return ni_benchmark(
    #     list_all_dataset, 
    #     list_all_dataset, 
    #     n_experiences=len(list_all_dataset), 
    #     shuffle=False, 
    #     balance_experiences=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)


    # return nc_benchmark(
    #     list_all_dataset,
    #     list_all_dataset,
    #     n_experiences=len(list_all_dataset),
    #     task_labels=True,
    #     shuffle=False,
    #     class_ids_from_zero_in_each_exp=True,
    #     one_dataset_per_exp=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)

    # return nc_benchmark(
    #     list_all_dataset,
    #     list_all_dataset,
    #     n_experiences=len(list_all_dataset),
    #     task_labels=True,
    #     shuffle=False,
    #     class_ids_from_zero_in_each_exp=True,
    #     one_dataset_per_exp=True,
    #     train_transform=train_transform,
    #     eval_transform=test_transform,
    #     seed=args.random_seed)
    
if __name__ == '__main__':
    dataset=get_data_set_online()
    import pdb;pdb.set_trace()
    print('finsih')
# # from torchvision.datasets import MNIST
# # from avalanche.benchmarks.datasets import default_dataset_location
# # dataset_root = default_dataset_location('mnist')
# # train_set = MNIST(root=dataset_root,
# #                       train=True, download=True)
# # import pdb;pdb.set_trace()


# from torchvision.datasets import MNIST
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
# train_transform = Compose([
#     RandomCrop(28, padding=4),
#     ToTensor(),
#     Normalize((0.1307,), (0.3081,))
# ])

# test_transform = Compose([
#     ToTensor(),
#     Normalize((0.1307,), (0.3081,))
# ])

# mnist_train = MNIST(
#     './data/mnist', train=True, download=True, transform=train_transform
# )
# mnist_test = MNIST(
#     './data/mnist', train=False, download=True, transform=test_transform
# )
# scenario = ni_benchmark(
#     mnist_train, mnist_test, n_experiences=10, shuffle=True, seed=1234,
#     balance_experiences=True
# )

# train_stream = scenario.train_stream

# for experience in train_stream:
#     t = experience.task_label
#     exp_id = experience.current_experience
#     training_dataset = experience.dataset
#     print('Task {} batch {} -> train'.format(t, exp_id))
#     print('This batch contains', len(training_dataset), 'patterns')
#     print("Current Classes: ", experience.classes_in_this_experience)
# [len(dataset.test_stream[ii].dataset) for ii in range(10)]
