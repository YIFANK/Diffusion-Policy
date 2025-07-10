import numpy as np
import torch
from tiny_embodied_reasoning.workspace import utils as utils
def preprocess(dataset_path):
    if dataset_path is None:
        return None
    scenes = utils.load_trajectories_pickle(dataset_path)
    # Initialize storage
    all_images = []
    all_states = []
    all_actions = []
    episode_ends = []
    frame_idx = 0
    for scene in scenes:
        # print(len(scene.trajectories))
        for traj in scene.trajectories:
            states = traj.data
            for state in states:
                obs_img = state.observation  # this should be shape (96, 96, 3)
                agent_pos = state.state['agent']['position'] # extract first two dims as agent_pos
                action = state.action      # (D,) action

                all_images.append(obs_img)
                all_states.append(agent_pos)
                all_actions.append(action)
                frame_idx += 1

            episode_ends.append(frame_idx)
    # Convert to numpy arrays
    all_images = np.stack(all_images, axis=0)   # (N, 96, 96, 3)

    all_states = np.stack(all_states, axis=0)   # (N, 2)
    all_actions = np.stack(all_actions, axis=0) # (N, D)
    all_images = all_images.astype(np.float32) / 255.0
    all_states = all_states.astype(np.float32)
    all_actions = all_actions.astype(np.float32)

    episode_ends = np.array(episode_ends)

    # Pack into dataset_root format
    dataset_root = {
        'data': {
            'img': all_images,
            'state': all_states,
            'action': all_actions
        },
        'meta': {
            'episode_ends': episode_ends
        }
    }
    return dataset_root

def rotate_2d_vector(vector):
    """rotate 2d vector on a 512 * 512 grid by 90 degrees"""
    x, y = vector
    x_new = 512 - y
    y_new = x
    return np.array([x_new, y_new])

def rotate_dataset(dataset_root, times):
    if dataset_root is None:
        return None
    """Rotate the dataset by 90 degrees `times` times, return new float32 dataset."""

    new_data = {
        'img': np.copy(dataset_root['data']['img']).astype(np.float32),
        'state': np.copy(dataset_root['data']['state']).astype(np.float32),
        'action': np.copy(dataset_root['data']['action']).astype(np.float32)
    }

    for _ in range(times % 4):
        new_data['img'] = np.rot90(new_data['img'], axes=(1, 2)).copy()
        new_data['state'] = np.array([rotate_2d_vector(s) for s in new_data['state']], dtype=np.float32)
        new_data['action'] = np.array([rotate_2d_vector(a) for a in new_data['action']], dtype=np.float32)

    return {'data': new_data, 'meta': dataset_root['meta']}


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_paths: list,
                 text_conditions: list,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 rotate: bool = False):
        
        assert len(dataset_paths) == len(text_conditions), "Each dataset must have a corresponding text condition."

        all_image_data = []
        all_agent_pos = []
        all_action = []
        all_episode_ends = []
        all_text_conditions = []

        total_offset = 0  # to track episode ends across datasets
        dataset_list = [preprocess(dataset_path) for dataset_path in dataset_paths]
        for i, text_cond in enumerate(text_conditions):
            if rotate:
                for j in range(4):
                    if dataset_list[j] is None:
                        continue
                    print(f'processing dataset {j}, length: {len(dataset_list[j]["data"]["img"])}')
                    dataset_root = rotate_dataset(dataset_list[j],i - j + 4)
                    # images
                    image_data = dataset_root['data']['img'][:]  # (N,96,96,3)
                    image_data = np.moveaxis(image_data, -1, 1)  # (N,3,96,96)
                    all_image_data.append(image_data)

                    # agent pos and action
                    state_data = dataset_root['data']['state'][:,:2]
                    action_data = dataset_root['data']['action'][:]
                    all_agent_pos.append(state_data)
                    all_action.append(action_data)

                    # episode ends
                    episode_ends = dataset_root['meta']['episode_ends'][:] + total_offset
                    all_episode_ends.append(episode_ends)
                    total_offset += state_data.shape[0]

                    # text conditions (repeat text_cond for each frame)
                    all_text_conditions.append([text_cond] * state_data.shape[0])
            else:
                if dataset_list[i] is None:
                    continue
                dataset_root = dataset_list[i]
                # images
                image_data = dataset_root['data']['img'][:]  # (N,96,96,3)
                image_data = np.moveaxis(image_data, -1, 1)  # (N,3,96,96)
                all_image_data.append(image_data)

                # agent pos and action
                state_data = dataset_root['data']['state'][:,:2]
                action_data = dataset_root['data']['action'][:]
                all_agent_pos.append(state_data)
                all_action.append(action_data)

                # episode ends
                episode_ends = dataset_root['meta']['episode_ends'][:] + total_offset
                all_episode_ends.append(episode_ends)
                total_offset += state_data.shape[0]

                # text conditions (repeat text_cond for each frame)
                all_text_conditions.append([text_cond] * state_data.shape[0])

        # concatenate everything
        all_image_data = np.concatenate(all_image_data, axis=0)
        all_agent_pos = np.concatenate(all_agent_pos, axis=0)
        all_action = np.concatenate(all_action, axis=0)
        all_episode_ends = np.concatenate(all_episode_ends, axis=0)
        all_text_conditions = sum(all_text_conditions, [])  # flatten list of lists

        train_data = {
            'agent_pos': all_agent_pos,
            'action': all_action
        }

        # compute indices
        indices = create_sample_indices(
            episode_ends=all_episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1
        )

        # compute normalization across the entire dataset
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images stay as they are
        normalized_train_data['image'] = all_image_data
        normalized_train_data['text'] = np.array(all_text_conditions)

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        nsample['image'] = nsample['image'][:self.obs_horizon, :]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
        nsample['text'] = nsample['text'][0]  # text is constant for entire sequence

        return nsample
import random
import sys
sys.path.append('../../')
from diffusion_policy.utils.visualization import visualize_trajectories
from diffusion_policy.utils.img_to_gif import images_to_gif
from PIL import Image

if __name__ == "__main__":
    Colors = ['Blue', 'Red', 'Green']
    colors = ['blue', 'red', 'green']
    #only use Blue for now
    path_list = [f'../../dataset/{color}_{num}.pkl' for color in Colors[:1] for num in [0,1,2,3]]
    path_list[2] = None
    path_list[3] = None
    description_list = [f'push the {color} block to the {num} corner' for color in colors[:1] for num in ['lower-right', 'upper-right', 'upper-left', 'lower-left']]
    dataset = PushTImageDataset(path_list,description_list, 
                                pred_horizon=40, obs_horizon=30,action_horizon=8,rotate = True)
    #visualize trajectories

    # actions = [dataset[i]['action'] for i in range(len(dataset)) if dataset[i]['text'] == 'push the blue block to the upper-right corner']
    # actions = np.stack(actions, axis=0)  # (N, pred_horizon, action_dim)
    # print(actions.shape)
    all_images = [dataset[i]['image'] for i in range(len(dataset)) if dataset[i]['text'] == 'push the blue block to the upper-left corner']
    #select 3 random image list
    images_list = [all_images[random.randint(0,len(all_images)-1)] for _ in range(3)]
    #images.shape: (10, 3, 96, 96)
    images_list = [image.transpose(0,2,3,1) for image in images_list]
    images_to_gif(images_list[0],save_path = '../../output/eval/images.gif',fps = 10)
    images_to_gif(images_list[1],save_path = '../../output/eval/images1.gif',fps = 10)
    images_to_gif(images_list[2],save_path = '../../output/eval/images2.gif',fps = 10)
    # visualize_trajectories(
    #     actions,
    #     n=50,
    #     gif_path="../../output/eval/trajectories.gif",
    #     fps=10,
    #     seed=42
    # )