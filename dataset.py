import torch
import torch.utils.data as data
import numpy as np
import os
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

category_list = ['amusement', 'anger', 'awe', 'disgust', 'contentment', 'excitement', 'fear', 'sadness']

class EEGDataset(data.Dataset):
    def __init__(self, eeg_dir, anno_dir, transform=None, is_classification=True, is_clip=False, is_all=False, is_staring=False, is_imagination=True):
        self.eeg_dir = eeg_dir
        self.transform = transform
        self.is_classification = is_classification
        self.is_clip = is_clip
        self.eeg_data = np.load(os.path.join(eeg_dir, 'eeg_data_array.npy'))
        self.is_all = is_all
        # self.eeg_data = torch.tensor(self.eeg_data, dtype=torch.float32)
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
        if self.is_all:
            return self.eeg_data.shape[0]
        else:
            return len(self.labels['stim_order'])

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


        if self.transform:
            eeg_sample = self.transform(eeg_sample)
        
        if self.is_classification:
            return eeg_sample, label
        
        if self.is_clip:
            return eeg_sample, outputs.last_hidden_state[0]
        else:
            return eeg_sample


if __name__=='__main__':
    import matplotlib.pyplot as plt
    eeg_dir = './datasets'
    anno_dir = './datasets/annotation_order.json'
    eeg_dataset = EEGDataset(eeg_dir, anno_dir, is_classification=False, is_clip=True)
    print(eeg_dataset[1][1].shape)
    # print(eeg_dataset[0][0].shape)
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
