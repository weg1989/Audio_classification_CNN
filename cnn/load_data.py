#!/usr/bin/env python
# coding: utf-8

# In[9]:


from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os
import torch


# # Steup device agnostic code
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device

# class UrbansoundDataset(Dataset):
#     
#     def __init__(self, annotation_file, audio_dir, transformation,target_sample_rate,num_samples):
#         self.annotations = pd.read_csv(annotation_file)
#         self.audio_dir = audio_dir
#         self.transformation = transformation
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = num_samples
#     
#     def __len__(self): # used to get length of an item
#         return len(self.annotations)
#     
#     def __getitem__(self, index): #a_list[1] -> a_list.__getitem__(1)
#         audio_sample_path = self._get_audio_sample_path(index)
#         label = self._get_audio_sample_label(index)
#         signal, sr = torchaudio.load(audio_sample_path)
#         signal = self._resample_if_necessary(signal, sr)
#         signal = self._mix_down_if_necessary(signal)
#         #signal = self._cut_if_necessary(signal)
#         #signal = self._right_pad_if_necessary(signal)
#         signal = self.transformation(signal)
#         return signal , label
#         
#     def _get_audio_sample_path(self, index):
#         fold = f"fold{self.annotations.iloc[index, 5]}"
#         path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
#         return path
#     
#     def _get_audio_sample_label(self, index):
#         return self.annotations.iloc[index, 6]
#     
#     def _resample_if_necessary(self, signal, sr):
#         if sr != self.target_sample_rate:
#             resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
#             signal = resampler(signal)
#         return signal
#     
#     def _mix_down_if_necessary(self, signal):
#         if signal.shape[0]>1:
#             signal = torch.mean(signal , dim=0, keepdim=True)
#         return signal
#     
# #     def _cut_if_necessary(self, signal):
# #         if signal.shape[1] > self.num_samples:
# #             signal = signal[:, :self.num_samples]
# #         return signal
#         
# #     def _right_pad_if_necessary(self, signal):
# #         length_signal = signal.shape[1]
# #         if length_signal < self.num_samples:
# #             num_missing_samples = self.num_samples - length_signal
# #             last_dim_padding = (0, num_missing_samples)  # 0 extra elements added to the left of tensor, num_missing_samples added to the right of tensor
# #             signal = torch.nn.functional.pad(signal, last_dim_padding)
# #         return signal

# Padding can be done to both left and right of a tensor as we have seen above 
# 
# It can also be done to different level of dimnesions
# (1,1,2,2) - this will add 1 at start and end of the outter most dimension
#             and 2 at start and end of the second outermost dimension

# In[14]:


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples
                 ):
        #self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


# In[15]:


SAMPLE_RATE=16000
NUM_SAMPLES = 22050

mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE,
                                              n_fft = 1024,
                                              hop_length = 512,
                                              n_mels = 64)


# In[16]:


ANNOTATIONS_FILE = r"E:\DL_audio\with_pytorch\data1\metadata\fold_1_annotation.csv"
AUDIO_DIR = r"E:\DL_audio\with_pytorch\data1\audio"
    
usd = UrbanSoundDataset(ANNOTATIONS_FILE,AUDIO_DIR,mel_spec,SAMPLE_RATE,NUM_SAMPLES)

print(f"There are {len(usd)} samples in the dataset.")

signal , label = usd[0]

print(signal) , print(label) , print(signal.shape), print(signal.device)

a=1


# In[ ]:





# In[ ]:




