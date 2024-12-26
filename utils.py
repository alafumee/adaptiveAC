import numpy as np
import torch
import torch.nn.functional as F
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

import torch.nn as nn
import torchvision.transforms as transforms
from detr.models.detr_vae import build_backbone
class Resnet(nn.Module):
    def __init__(self,args):
        super().__init__()
        backbones = []
        backbone = build_backbone(args)
        backbones = [backbone]
        self.backbones = nn.ModuleList(backbones)
        print(self.backbones[0].num_channels,end=" num channels ???????????????????????????\n")

    def forward(self, x):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        x = normalize(x)
        x=x.squeeze(0)
        x,_=self.backbones[0](x)
        # print(x[0].shape,end="   x shape\n")
        x=x[0]
        return x.unsqueeze(0)

class EpisodicDataset_prediction(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, args):
        super(EpisodicDataset_prediction).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None

        self.resnet = Resnet(args).cuda()
        self.episode_len = 400
        from tqdm import tqdm
        for episode_id in tqdm(episode_ids):
            print(f"Processing episode {episode_id}")
            if os.path.exists(os.path.join(self.dataset_dir, f'features_episode_{episode_id}.hdf5')):
                continue
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            feature_file_path = os.path.join(self.dataset_dir, f'features_episode_{episode_id}.hdf5')

            with h5py.File(dataset_path, 'r') as root, h5py.File(feature_file_path, 'w') as f_out:
                # Create a dataset to store the features
                feature_shape = (self.episode_len, len(self.camera_names), self.resnet.backbones[0].num_channels)
                feature_dataset = f_out.create_dataset('features', shape=feature_shape, dtype=np.float32)

                for i in range(self.episode_len):
                    image_data = self.load_and_preprocess_image(root, i)
                    with torch.no_grad():  # Disable gradient computation
                        image_rep = self.resnet(image_data)
                    image_rep = torch.amax(image_rep, dim=(-2, -1))
                    # Normalize the image representation to have L2 norm of dim-1
                    dim = image_rep.shape[-1]
                    image_rep = F.normalize(image_rep, p=2, dim=-1) * (dim)**0.5
                    feature_dataset[i] = image_rep.cpu().numpy()

                    if i % 50 == 0:
                        print(f"Processed {i}/{self.episode_len} frames")

                    # Clear CUDA cache
                    torch.cuda.empty_cache()

                # Add attributes to store metadata
                f_out.attrs['episode_id'] = episode_id
                f_out.attrs['feature_shape'] = feature_shape
                f_out.attrs['camera_names'] = self.camera_names

            print(f"Features saved to {feature_file_path}")

        self.__getitem__(0)  # initialize self.is_sim

    def load_and_preprocess_image(self, root, index):
        image_dict = {cam_name: root[f'/observations/images/{cam_name}'][index:index+1]
                      for cam_name in self.camera_names}
        all_cam_images = np.stack([image_dict[cam_name] for cam_name in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images).float() / 255.0
        image_data = torch.einsum('k t h w c -> k t c h w', image_data).cuda()
        return image_data

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        feature_file_path = os.path.join(self.dataset_dir, f'features_episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root, h5py.File(feature_file_path, 'r') as f_in:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            # print(self.camera_names,end=" camera names \n")
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts:start_ts+1]
            features = f_in['features'][start_ts:]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_features = np.zeros((self.episode_len,len(self.camera_names),self.resnet.backbones[0].num_channels),dtype=np.float32)
        padded_action[:action_len] = action
        padded_features[:action_len] = features
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        features_data = torch.from_numpy(padded_features)
        image_data = image_data / 255.0
        # channel last
        # print(image_data.shape)
        image_data = torch.einsum('k t h w c -> k t c h w', image_data)
        image_data = image_data[:,0,:,:,:]
        # image_rep = self.resnet(image_data)
        # image_rep = torch.amax(image_rep, dim=(-2, -1))

        # normalize image and change dtype to float

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # print(image_data.shape,features_data.shape,qpos_data.shape,action_data.shape,end=" output shape \n")

        return image_data, features_data, qpos_data, action_data, is_pad

class EpisodicDataset_MINE(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        # padded_action = np.zeros(original_action_shape, dtype=np.float32)
        # padded_action[:action_len] = action
        # is_pad = np.zeros(episode_len)
        # is_pad[action_len:] = 1
        # action_interval = np.random.randint(0, min(self.chunk_size, action_len))
        action_interval = np.random.randint(0, self.chunk_size)
        if action_interval < action_len:   
            action = action[action_interval]
        else:
            action = np.zeros_like(action[-1])

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action).float()
        # is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, action_interval #, is_pad

# class EpisodicDatasetPredictionFullFeature(torch.utils.data.Dataset):
#     def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, args):
#         super().__init__()
#         self.episode_ids = episode_ids
#         self.dataset_dir = dataset_dir
#         self.camera_names = camera_names
#         self.norm_stats = norm_stats
#         self.is_sim = None

#         self.resnet = Resnet(args).cuda()
#         self.episode_len = 400
#         from tqdm import tqdm
#         for episode_id in tqdm(episode_ids):
#             print(f"Processing episode {episode_id}")
#             if os.path.exists(os.path.join(self.dataset_dir, f'resnet_features_episode_{episode_id}.hdf5')):
#                 continue
#             dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
#             feature_file_path = os.path.join(self.dataset_dir, f'resnet_features_episode_{episode_id}.hdf5')

#             with h5py.File(dataset_path, 'r') as root, h5py.File(feature_file_path, 'w') as f_out:
#                 # Create a dataset to store the features
#                 feature_shape = (self.episode_len, len(self.camera_names), self.resnet.backbones[0].num_channels)
#                 feature_dataset = f_out.create_dataset('features', shape=feature_shape, dtype=np.float32)

#                 for i in range(self.episode_len):
#                     image_data = self.load_and_preprocess_image(root, i)
#                     with torch.no_grad():  # Disable gradient computation
#                         image_rep = self.resnet(image_data)
#                         image_rep = torch.reshape(image_rep, (image_rep.shape[0], image_rep.shape[1], -1))
#                     # image_rep = torch.amax(image_rep, dim=(-2, -1))
#                     feature_dataset[i] = image_rep.cpu().numpy()

#                     if i % 50 == 0:
#                         print(f"Processed {i}/{self.episode_len} frames")

#                     # Clear CUDA cache
#                     torch.cuda.empty_cache()

#                 # Add attributes to store metadata
#                 f_out.attrs['episode_id'] = episode_id
#                 f_out.attrs['feature_shape'] = feature_shape
#                 f_out.attrs['camera_names'] = self.camera_names

#             print(f"Features saved to {feature_file_path}")

#         self.__getitem__(0)  # initialize self.is_sim

#     def load_and_preprocess_image(self, root, index):
#         image_dict = {cam_name: root[f'/observations/images/{cam_name}'][index:index+1]
#                       for cam_name in self.camera_names}
#         all_cam_images = np.stack([image_dict[cam_name] for cam_name in self.camera_names], axis=0)
#         image_data = torch.from_numpy(all_cam_images).float() / 255.0
#         image_data = torch.einsum('k t h w c -> k t c h w', image_data).cuda()
#         return image_data

#     def __len__(self):
#         return len(self.episode_ids)

#     def __getitem__(self, index):
#         sample_full_episode = False # hardcode

#         episode_id = self.episode_ids[index]
#         dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
#         feature_file_path = os.path.join(self.dataset_dir, f'features_episode_{episode_id}.hdf5')
#         with h5py.File(dataset_path, 'r') as root, h5py.File(feature_file_path, 'r') as f_in:
#             is_sim = root.attrs['sim']
#             original_action_shape = root['/action'].shape
#             episode_len = original_action_shape[0]
#             if sample_full_episode:
#                 start_ts = 0
#             else:
#                 start_ts = np.random.choice(episode_len)
#             # get observation at start_ts only
#             qpos = root['/observations/qpos'][start_ts]
#             qvel = root['/observations/qvel'][start_ts]
#             image_dict = dict()
#             # print(self.camera_names,end=" camera names \n")
#             for cam_name in self.camera_names:
#                 image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts:start_ts+1]
#             features = f_in['features'][start_ts:]
#             # get all actions after and including start_ts
#             if is_sim:
#                 action = root['/action'][start_ts:]
#                 action_len = episode_len - start_ts
#             else:
#                 action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
#                 action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

#         self.is_sim = is_sim
#         padded_action = np.zeros(original_action_shape, dtype=np.float32)
#         padded_features = np.zeros((self.episode_len,len(self.camera_names),self.resnet.backbones[0].num_channels),dtype=np.float32)
#         padded_action[:action_len] = action
#         padded_features[:action_len] = features
#         is_pad = np.zeros(episode_len)
#         is_pad[action_len:] = 1

#         # new axis for different cameras
#         all_cam_images = []
#         for cam_name in self.camera_names:
#             all_cam_images.append(image_dict[cam_name])
#         all_cam_images = np.stack(all_cam_images, axis=0)

#         # construct observations
#         image_data = torch.from_numpy(all_cam_images)
#         qpos_data = torch.from_numpy(qpos).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()
#         features_data = torch.from_numpy(padded_features)
#         image_data = image_data / 255.0
#         # channel last
#         # print(image_data.shape)
#         image_data = torch.einsum('k t h w c -> k t c h w', image_data)
#         image_data = image_data[:,0,:,:,:]
#         # image_rep = self.resnet(image_data)
#         # image_rep = torch.amax(image_rep, dim=(-2, -1))

#         # normalize image and change dtype to float

#         action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
#         qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

#         # print(image_data.shape,features_data.shape,qpos_data.shape,action_data.shape,end=" output shape \n")

#         return image_data, features_data, qpos_data, action_data, is_pad



def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    import argparse
    args = argparse.Namespace()
    args.hidden_dim = 512  # You can adjust this value as needed
    args.position_embedding = 'sine'
    args.camera_names = camera_names
    args.lr_backbone = 0.0
    args.dilation = False
    args.masks = False
    args.backbone = 'resnet18'


    train_dataset_prediction = EpisodicDataset_prediction(train_indices, dataset_dir, camera_names, norm_stats,args)
    val_dataset_prediction = EpisodicDataset_prediction(val_indices, dataset_dir, camera_names, norm_stats,args)
    train_dataloader_prediction = DataLoader(train_dataset_prediction, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader_prediction = DataLoader(val_dataset_prediction, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    return train_dataloader, val_dataloader, train_dataloader_prediction, val_dataloader_prediction, norm_stats, train_dataset.is_sim

def load_data_MINE(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, mine_batch_size, chunk_size):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    train_dataset_prediction = EpisodicDataset_MINE(train_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    val_dataset_prediction = EpisodicDataset_MINE(val_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    train_dataloader_prediction = DataLoader(train_dataset_prediction, batch_size=mine_batch_size, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader_prediction = DataLoader(val_dataset_prediction, batch_size=mine_batch_size, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    return train_dataloader, val_dataloader, train_dataloader_prediction, val_dataloader_prediction, norm_stats, train_dataset.is_sim

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    print("epoch_dicts: ", epoch_dicts)
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


