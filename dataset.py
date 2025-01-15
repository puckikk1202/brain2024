import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import os
import json
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import butter, filtfilt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import math

category_list = ['amusement', 'excitement', 'awe', 'contentment', 'disgust', 'anger', 'fear', 'sadness']

class EEGDataset(data.Dataset):
    def __init__(self, eeg_dir, anno_dir, transform=None, is_classification=True, is_binary=False, is_clip=False, is_all=False, is_staring=False, is_imagination=True, compute_de_psd=False):
        self.eeg_dir = eeg_dir
        self.transform = transform
        self.is_classification = is_classification
        self.is_clip = is_clip
        self.eeg_data = np.load(os.path.join(eeg_dir, 'takahashi_data.npy')) # eeg_data_array.npy
        # self.eeg_data = np.load(os.path.join(eeg_dir, 'eeg_data_array.npy')) # eeg_data_array.npy
        self.is_all = is_all
        self.compute_de_psd = compute_de_psd
        self.is_binary = is_binary
        # self.eeg_data = torch.tensor(self.eeg_data, dtype=torch.float32)
        if self.compute_de_psd:
            self.attn_merge = MutualCrossAttention(0.3)
            self.de_values = compute_de(self.eeg_data[:,:,3*512:6*512])
            self.psd_values = compute_psd(self.eeg_data[:,:,3*512:6*512])
            bands_counter = []
            de_bands_counter = []
            psd_bands_counter = []
            for i in range(5):
                single_de = self.de_values[:, :, i, :]
                single_psd = self.psd_values[:, :, i, :]
                band = self.attn_merge(torch.tensor(single_de), torch.tensor(single_psd))
                # print('band', band)
                bands_counter.append(band)
                de_bands_counter.append(torch.tensor(single_de))
                psd_bands_counter.append(torch.tensor(single_psd))

            self.mca_values = torch.stack(bands_counter, dim=2)
            self.mca_values = self.mca_values.unsqueeze(1)
            self.mca_values = (self.mca_values - torch.min(self.mca_values) / (torch.max(self.mca_values) - torch.min(self.mca_values))) * 2 - 1
            # self.mca_values = self.attn_merge(torch.tensor(self.de_values), torch.tensor(self.psd_values))
            # self.mca_values = self.normalize_eeg_data(self.mca_values)
            # print('mca', self.mca_values)
            self.de_values = torch.stack(de_bands_counter, dim=2)
            self.psd_values = torch.stack(psd_bands_counter, dim=2)
            # self.de_values = self.de_values.unsqueeze(1)
            # self.psd_values = self.psd_values.unsqueeze(1)

        if is_staring:
            self.eeg_data = self.normalize_eeg_data(self.eeg_data)
            self.eeg_data = self.eeg_data[:,:,3*512:6*512]
        elif is_imagination:
            self.eeg_data = self.normalize_eeg_data(self.eeg_data)
            self.eeg_data = self.eeg_data[:,:,6*512:9*512]
        elif is_all:
            self.eeg_data = self.normalize_eeg_data(self.eeg_data)
            self.eeg_data = self.eeg_data.reshape(self.eeg_data.shape[0], 64, 3, 1536)
            self.eeg_data = self.eeg_data.transpose(0, 2, 1, 3).reshape(self.eeg_data.shape[0]*3, 64, 1536)

        if is_clip:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14") #clip-vit-large-patch14
        # self.labels = self.load_labels()
        self.anno_dir = anno_dir
        with open(self.anno_dir, 'r') as f:
            self.labels = json.load(f)

    def normalize_eeg_data(self, eeg_data):
        """Normalize each channel globally to follow a standard normal distribution."""
        # Assuming eeg_data is in the shape [num_samples, num_channels]
        num_channels = eeg_data.shape[1]
        
        # Normalize each channel separately
        for channel in range(num_channels):
            channel_mean = np.mean(eeg_data[:, channel])  # Calculate mean for this channel
            channel_std = np.std(eeg_data[:, channel])    # Calculate std for this channel
            eeg_data[:, channel] = (eeg_data[:, channel] - channel_mean) / channel_std  # Normalize
        
        return eeg_data

    def __len__(self):
        # if self.is_all:
            return self.eeg_data.shape[0]
        # else:
            # return len(self.labels['stim_order'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        eeg_sample = self.eeg_data[idx]
        
        if self.is_clip:
            image = Image.open('./datasets/images/' + self.labels['stim_order'][idx])
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
            # eeg_sample = torch.cat((eeg_sample, outputs.last_hidden_state[0]), dim=0)


        if self.is_classification:
            label = self.labels['stim_order'][idx]
            label = label.split('/')[0]
            category = category_list.index(label)
            one_hot_label = torch.zeros(len(category_list), dtype=torch.float32)
            one_hot_label[category] = 1.0
            label = one_hot_label

        elif self.is_binary:
            label = self.labels['stim_order'][idx]
            label = label.split('/')[0]
            category = category_list.index(label)
            one_hot_label = torch.zeros(2, dtype=torch.float32)
            if category < 4:
                one_hot_label[0] = 1.0
            else:
                one_hot_label[1] = 1.0
            label = one_hot_label

        if self.transform:
            eeg_sample = self.transform(eeg_sample)
        
        # if self.is_classification:
            # return eeg_sample, outputs.last_hidden_state[0], label
        
        if self.is_clip or self.is_classification or self.is_binary:
            output = {'eeg_sample': eeg_sample, 
                      'mca_feat': self.mca_values[idx], 
                      'clip_feat': outputs.last_hidden_state[0], 
                      'label': label,
                      'de_feat': self.de_values[idx],
                      'psd_feat': self.psd_values[idx]} 
            return output
            # return eeg_sample, self.mca_values[idx], outputs.last_hidden_state[0], label, self.de_values[idx], self.psd_values[idx]
        # elif self.compute_de_psd:
            # return eeg_sample, self.de_values[idx], self.psd_values[idx], outputs.last_hidden_state[0], label
        else:
            return eeg_sample

def compute_psd(data, fs=512, window_sec=3, segment_sec=1): 
    """
    计算每个样本、每个通道、每个时间窗口和每个频段的 PSD 特征。

    参数:
    - data: numpy 数组，形状为 (samples, channels, samples_in_channel)
    - fs: 采样频率
    - window_sec: Welch 方法中的窗口长度，以秒为单位
    - segment_sec: 划分时间窗口的长度，以秒为单位

    返回:
    - psd_values: 每个样本的 PSD 特征，形状为 (samples, channels, frequency_bands, time_windows)
    """
    nperseg = int(window_sec * fs)  # Welch 方法中的窗口大小
    segment_size = int(segment_sec * fs)  # 每个时间窗口的长度
    freq_bands = {
        'theta': (4, 7),
        'alpha': (8, 10),
        'slow_alpha': (8, 13),
        'beta': (14, 29),
        'gamma': (30, 45)
    }

    samples, channels, samples_in_channel = data.shape
    time_windows = samples_in_channel // segment_size  # 计算完整的时间窗口数
    
    # 初始化 PSD 特征数组
    psd_values = np.zeros((samples, channels, len(freq_bands), time_windows))
    
    print('computing PSD values...')
    for s in tqdm(range(samples)):
        for ch in range(channels):
            for t in range(time_windows):
                # 取出每个时间窗口的数据
                window_data = data[s, ch, t * segment_size:(t + 1) * segment_size]
                
                # 计算 Welch 的 PSD
                f, psd = welch(window_data, fs=fs, nperseg=nperseg, window='hann')
                band_psd = []
                
                # 计算每个频段的 PSD 平均值
                for low, high in freq_bands.values():
                    idx_band = np.logical_and(f >= low, f <= high)
                    band_psd.append(np.mean(psd[idx_band]))
                
                # 存储当前时间窗口的频段 PSD
                psd_values[s, ch, :, t] = band_psd
    
    return psd_values

def compute_de(data, fs=512, window_sec=3, segment_sec=1):
    """
    计算每个样本、每个通道、每个时间窗口和每个频段的 DE 特征。

    参数:
    - data: numpy 数组，形状为 (samples, channels, samples_in_channel)
    - fs: 采样频率
    - window_sec: 计算 DE 的窗口长度，以秒为单位
    - segment_sec: 划分时间窗口的长度，以秒为单位

    返回:
    - de_values: 每个样本的 DE 特征，形状为 (samples, channels, frequency_bands, time_windows)
    """
    window_size = int(window_sec * fs)  # DE 计算窗口大小
    segment_size = int(segment_sec * fs)  # 每个时间窗口的长度
    freq_bands = {
        'theta': (4, 7),
        'alpha': (8, 10),
        'slow_alpha': (8, 13),
        'beta': (14, 29),
        'gamma': (30, 45)
    }

    samples, channels, samples_in_channel = data.shape
    time_windows = samples_in_channel // segment_size  # 计算完整的时间窗口数
    
    # 初始化 DE 特征数组
    de_values = np.zeros((samples, channels, len(freq_bands), time_windows))
    
    print('computing DE values...')
    for s in tqdm(range(samples)):
        for ch in range(channels):
            for t in range(time_windows):
                # 取出每个时间窗口的数据
                window_data = data[s, ch, t * segment_size:(t + 1) * segment_size]
                
                de_band = []
                # 对每个频段计算 DE
                for low, high in freq_bands.values():
                    # 使用带通滤波器提取该频段的数据
                    band_data = bandpass_filter(window_data, low, high, fs)
                    
                    # 检查 band_data 是否足够长
                    if len(band_data) < window_size:
                        # 如果长度不足，则将 window_size 设置为 band_data 的长度
                        current_window_size = len(band_data)
                    else:
                        current_window_size = window_size
                    
                    # 计算该频段内的 DE
                    de_channel = []
                    for start in range(0, len(band_data) - current_window_size + 1, current_window_size):
                        segment_data = band_data[start:start + current_window_size]
                        std_dev = np.std(segment_data)
                        de = 0.5 * np.log(2 * np.pi * np.e * (std_dev ** 2))
                        de_channel.append(de)
                        # print(de)
                    
                    # 取该频段的平均 DE 值
                    de_band.append(np.mean(de_channel) if de_channel else np.nan)
                    # print(de_band)
                
                # 存储当前时间窗口的频段 DE
                de_values[s, ch, :, t] = de_band
    
    return de_values

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

class MutualCrossAttention(nn.Module):
    def __init__(self, dropout):
        super(MutualCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # Assign x1 and x2 to query and key
        query = x1
        key = x2
        d = query.shape[-1]

        # Basic attention mechanism formula to get intermediate output A
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        output_A = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x2)
        # Basic attention mechanism formula to get intermediate output B
        scores = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(d)
        output_B = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x1)

        # Make the summation of the two intermediate outputs
        output = output_A + output_B  # shape (1280, 32, 60)

        return output

if __name__=='__main__':
    import matplotlib.pyplot as plt
    eeg_dir = './datasets'
    anno_dir = './datasets/annotation_order.json'
    eeg_dataset = EEGDataset(eeg_dir, anno_dir, is_staring=True, is_classification=False, is_binary=True, is_clip=True, compute_de_psd=True)
    attn_merge = MutualCrossAttention(0.3)

    print(eeg_dataset[0]['mca_feat'].shape)
    print(eeg_dataset[0]['clip_feat'].shape)
    print(eeg_dataset[0]['label'])
    print(eeg_dataset[0]['de_feat'].shape)
    print(eeg_dataset[0]['psd_feat'].shape)


    # x1, x2 = torch.tensor(eeg_dataset[0][1]).unsqueeze(0), torch.tensor(eeg_dataset[0][2]).unsqueeze(0)  
    # output = attn_merge(x1, x2)
    # print(output.shape) 
    # print(eeg_dataset[0][1])

    # fig = plt.figure()
    # plt.figure(figsize=(20, 5))
    # plt.imshow(eeg_dataset[0][0][:,:256], cmap='gray', aspect='auto')
    # plt.colorbar()
    # plt.title('Single-Channel Image Representation')
    # plt.xlabel('Width (1536)')
    # plt.ylabel('Height (64)')

    # Save the figure as a PNG image
    # file_path = 'single_channel_image.png'
    # plt.savefig(file_path)
