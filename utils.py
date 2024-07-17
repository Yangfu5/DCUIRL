import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
import math


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    if os.path.exists(dir_path):
        import shutil
        shutil.rmtree(dir_path)
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None,
                 mtm_bsz=64,
                 mtm_length=10,
                 mtm_ratio=0.15
                 ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.mtm_ratio = mtm_ratio
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.mtm_bsz = mtm_bsz
        self.mtm_length = mtm_length

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        # pos = obses.copy()

        # 对所有的观测进行裁减
        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        # pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        # pos = torch.as_tensor(pos, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None)
        # return obses, actions, rewards, next_obses, not_dones, cpc_kwargs
        return obses, actions, rewards, next_obses, not_dones

    def sample_ctmr(self):
        idxs = np.random.randint(
            0, self.capacity - self.mtm_length if self.full else self.idx - self.mtm_length,
            size=self.mtm_bsz
        )
        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.mtm_length).reshape(1, -1)
        idxs = idxs + step
        obses_label = self.obses[idxs]
        non_masked = np.zeros((self.mtm_bsz, self.mtm_length), dtype=bool)

        obses = self.random_obs(obses_label, non_masked)
        non_masked = torch.as_tensor(non_masked, device=self.device)
        obses_label = random_crop_2(obses_label, self.image_size)
        obses = random_crop_2(obses, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        obses_label = torch.as_tensor(obses_label, device=self.device).float()

        return (*self.sample_cpc(), dict(obses=obses, obses_label=obses_label, non_masked=non_masked))

    def sample_curl(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def random_obs(self, obses, non_masked):
        masked_obses = np.array(obses, copy=True)
        for row in range(self.mtm_bsz):
            for col in range(self.mtm_length):
                prob = random.random()
                if prob < self.mtm_ratio:
                    prob /= self.mtm_ratio
                    if prob < 0.8:
                        masked_obses[row, col] = 0
                    elif prob < 0.9:
                        masked_obses[row, col] = self.get_random_obs()
                    non_masked[row, col] = True
        return masked_obses

    def random_act(self, actions, non_masked):
        masked_actions = np.array(actions, copy=True)
        for row in range(self.mtm_bsz):
            for col in range(self.mtm_length):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        masked_actions[row, col] = 0
                    elif prob < 0.9:
                        masked_actions[row, col] = self.get_random_act()
                    non_masked[row, col + self.mtm_length + 1] = True
        return masked_actions

    def get_random_act(self):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        return self.actions[idx]

    def get_random_obs(self):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        obs = self.obses[idx]
        return obs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def save_last_size(self, max_size, save_dir=None, save_path=None):
        if self.idx == self.last_save:
            print('No buffer saved.')
            return
        pre_i = 0 if (self.idx - max_size < 0) else self.idx - max_size
        if save_path is None and save_dir is not None:
            save_path = f'{save_dir}/buffer[{pre_i}:{self.idx}]'
        if save_path is None:
            print('No path is provided!')
            return
        payload = [
            self.obses[pre_i: self.idx],
            self.next_obses[pre_i: self.idx],
            self.actions[pre_i: self.idx],
            self.rewards[pre_i: self.idx],
            self.not_dones[pre_i: self.idx],
        ]
        print(f'Try to save buffer[{pre_i}:{self.idx}] in {save_path}.')
        torch.save(payload, save_path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


def load_expert_buffer(file_path, buffer_size, batch_size, device, image_size=84, transform=None) -> ReplayBuffer:
    print(file_path)
    expert_data = torch.load(file_path)

    capacity = min(expert_data[0].shape[0], buffer_size)
    obs_space = expert_data[0].shape[1:]
    act_space = expert_data[2].shape[1:]

    buffer = ReplayBuffer(obs_space, act_space, capacity, batch_size, device, image_size, transform)

    buffer.obses[:] = expert_data[0][:capacity]
    buffer.next_obses[:] = expert_data[1][:capacity]
    buffer.actions[:] = expert_data[2][:capacity]
    buffer.rewards[:] = expert_data[3][:capacity]
    buffer.not_dones[:] = expert_data[4][:capacity]
    buffer.full = True  # 设置buffer已经满了，方便后续采样
    return buffer


# assume every episode have max_episode_steps


def get_buffer_return_mean_std(buffer, max_episode_steps=125):
    totla_steps = buffer.rewards.shape[0]
    ep_nums = totla_steps // max_episode_steps
    ep_returns = np.empty((ep_nums), np.float32)
    for ep_i in range(ep_nums):
        start_i = ep_i * max_episode_steps
        ep_returns[ep_i] = buffer.rewards[start_i:start_i + max_episode_steps, 0].sum()
    return_mean = float(ep_returns.mean())
    return_std = float(ep_returns.std())
    return return_mean, return_std


def load_model_sample_buffer(work_dir, load_step, buffer_size):
    import json
    from train import make_agent
    from collections import namedtuple
    import dmc2gym

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: " + str(device))

    with open(f"{work_dir}/args.json") as fp:
        args_dict = json.load(fp)
        # 使用namedtuple进行字典转对象
        Args = namedtuple("Args", args_dict.keys())
        args = Args(**args_dict)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )

    env.seed(args.seed)

    if args.encoder_type == 'pixel':
        env = FrameStack(env, k=args.frame_stack)

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (
            3 * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(obs_shape, action_shape, args, device)

    agent.load(f"{work_dir}/model", load_step)

    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=buffer_size + 4,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    # start sample
    with torch.no_grad():
        done = True
        episode_reward = 0
        episode_step = 0
        episode = 0
        for i in range(buffer_size):
            if done:
                obs = env.reset()  # 原来状态的维度为 (9, 100, 100)
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            with eval_mode(agent):
                action = agent.sample_action(obs)
            next_obs, reward, done, _ = env.step(action)

            # allow infinit bootstrap
            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
                done
            )
            episode_reward += reward
            replay_buffer.add(obs, action, reward, next_obs, done_bool)

            obs = next_obs
            episode_step += 1

        mean, std = get_buffer_return_mean_std(replay_buffer, env._max_episode_steps)
        buffer_save_path = f"{work_dir}/buffer/R[{int(mean)}&{int(std)}]S[{buffer_size}].pt"
        replay_buffer.save_last_size(
            buffer_size, save_path=buffer_save_path)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def random_crop_2(imgs, output_size):
    n1 = imgs.shape[0]
    n2 = imgs.shape[1]
    n = n1 * n2
    imgs = imgs.reshape(n, *imgs.shape[2:])
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs.reshape(n1, n2, *cropped_imgs.shape[1:])


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def rgb_to_grey(rgb_obs):
    pass


def sigmoid_cross_entropy_with_logits(logits: torch.Tensor, label, weight: torch.Tensor = None):
    zeros = torch.zeros_like(logits)
    cond = (logits >= zeros)
    relu_logits = torch.where(cond, logits, zeros)
    neg_abs_logits = torch.where(cond, -logits, logits)
    res = (relu_logits - logits * label) + torch.log1p(torch.exp(neg_abs_logits))
    if weight is not None:
        res *= weight
    return res.mean()


def control_cpu(cpu_num=4):
    import os
    import torch

    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
