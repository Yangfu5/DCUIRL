# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from utils import random_crop
# import time
# import os
#
# '''
# description: 具有固定大小的循环存储的buffer
# return {*}
# '''
#
#
# class ReplayBuffer(Dataset):
#     """Buffer to store environment transitions."""
#
#     def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None):
#         self.capacity = capacity
#         self.batch_size = batch_size
#         self.device = device
#         self.image_size = image_size
#         self.transform = transform
#         # the proprioceptive obs is stored as float32, pixels obs as uint8
#         obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
#
#         self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
#         self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
#         self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
#         self.rewards = np.empty((capacity, 1), dtype=np.float32)
#         self.not_dones = np.empty((capacity, 1), dtype=np.float32)
#
#         self.idx = 0
#         self.last_save = 0
#         self.full = False
#
#     def add(self, obs, action, reward, next_obs, done):
#         np.copyto(self.obses[self.idx], obs)
#         np.copyto(self.actions[self.idx], action)
#         np.copyto(self.rewards[self.idx], reward)
#         np.copyto(self.next_obses[self.idx], next_obs)
#         np.copyto(self.not_dones[self.idx], not done)
#
#         self.idx = (self.idx + 1) % self.capacity
#         self.full = self.full or self.idx == 0
#
#     def sample_proprio(self):
#
#         idxs = np.random.randint(
#             0, self.capacity if self.full else self.idx, size=self.batch_size
#         )
#
#         obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
#         actions = torch.as_tensor(self.actions[idxs], device=self.device)
#         rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
#         next_obses = torch.as_tensor(
#             self.next_obses[idxs], device=self.device).float()
#         not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
#         return obses, actions, rewards, next_obses, not_dones
#
#     '''
#     description: 对比预测编码CPC采样方法, 通过对状态图像进行随机裁剪, 得到pos
#     param {*} self
#     return {*}
#     '''
#
#     def sample_cpc(self):
#
#         start = time.time()
#         idxs = np.random.randint(
#             0, self.capacity if self.full else self.idx, size=self.batch_size
#         )
#         # 观测
#         obses = self.obses[idxs]
#         next_obses = self.next_obses[idxs]
#         # 观测对应的正样本
#         pos = obses.copy()
#         # ? 对所有观测进行随机裁剪 [B, C, 100, 100] -> [B, C, 84, 84]
#         obses = random_crop(obses, self.image_size)
#         next_obses = random_crop(next_obses, self.image_size)
#         pos = random_crop(pos, self.image_size)
#
#         obses = torch.as_tensor(obses, device=self.device).float()
#         next_obses = torch.as_tensor(next_obses, device=self.device).float()
#         pos = torch.as_tensor(pos, device=self.device).float()
#         actions = torch.as_tensor(self.actions[idxs], device=self.device)
#         rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
#         not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
#
#         cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None)
#
#         return obses, actions, rewards, next_obses, not_dones, cpc_kwargs
#
#     def save(self, save_dir):
#         if self.idx == self.last_save:
#             print("Not save any buffer.")
#             return
#         path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
#         payload = [
#             self.obses[self.last_save:self.idx],
#             self.next_obses[self.last_save:self.idx],
#             self.actions[self.last_save:self.idx],
#             self.rewards[self.last_save:self.idx],
#             self.not_dones[self.last_save:self.idx]
#         ]
#         self.last_save = self.idx
#         torch.save(payload, path)
#
#     def save_last_size(self, max_size, save_dir=None, save_path=None):
#         if self.idx == self.last_save:
#             print("No buffer saved.")
#             return
#         pre_i = 0 if (self.idx - max_size < 0) else self.idx - max_size
#         if save_path is None and save_dir is not None:
#             save_path = f"{save_dir}/buffer[{pre_i}:{self.idx}]"
#         if save_path is None:
#             print("No path is provided!")
#             return
#         payload = [
#             self.obses[pre_i:self.idx],
#             self.next_obses[pre_i:self.idx],
#             self.actions[pre_i:self.idx],
#             self.rewards[pre_i:self.idx],
#             self.not_dones[pre_i:self.idx]
#         ]
#         print(f"Try to save buffer[{pre_i}:{self.idx}] in {save_path}.")
#         torch.save(payload, save_path)
#
#     def load(self, save_dir):
#         chunks = os.listdir(save_dir)
#         chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
#         for chunk in chucks:
#             start, end = [int(x) for x in chunk.split('.')[0].split('_')]
#             path = os.path.join(save_dir, chunk)
#             payload = torch.load(path)
#             assert self.idx == start
#             self.obses[start:end] = payload[0]
#             self.next_obses[start:end] = payload[1]
#             self.actions[start:end] = payload[2]
#             self.rewards[start:end] = payload[3]
#             self.not_dones[start:end] = payload[4]
#             self.idx = end
#
#     def __getitem__(self, idx):
#         idx = np.random.randint(
#             0, self.capacity if self.full else self.idx, size=1
#         )
#         idx = idx[0]
#         obs = self.obses[idx]
#         action = self.actions[idx]
#         reward = self.rewards[idx]
#         next_obs = self.next_obses[idx]
#         not_done = self.not_dones[idx]
#
#         if self.transform:
#             obs = self.transform(obs)
#             next_obs = self.transform(next_obs)
#
#         return obs, action, reward, next_obs, not_done
#
#     def __len__(self):
#         return self.capacity
#
#
# def load_expert_buffer(file_path, buffer_size, batch_size, device, image_size=84, transform=None) -> ReplayBuffer:
#     expert_data = torch.load(file_path)
#
#     capacity = min(expert_data[0].shape[0], buffer_size)
#     obs_space = expert_data[0].shape[1:]
#     act_space = expert_data[2].shape[1:]
#
#     buffer = ReplayBuffer(obs_space, act_space, capacity,
#                           batch_size, device, image_size, transform)
#
#     buffer.obses[:] = expert_data[0][:capacity]
#     buffer.next_obses[:] = expert_data[1][:capacity]
#     buffer.actions[:] = expert_data[2][:capacity]
#     buffer.rewards[:] = expert_data[3][:capacity]
#     buffer.not_dones[:] = expert_data[4][:capacity]
#     buffer.full = True  # 设置buffer已经满了，方便后续采样
#     return buffer
#
# # assume every episode have max_episode_steps
#
#
# def get_buffer_return_mean_std(buffer, max_episode_steps=125):
#     totla_steps = buffer.rewards.shape[0]
#     ep_nums = totla_steps // max_episode_steps
#     ep_returns = np.empty((ep_nums), np.float32)
#     for ep_i in range(ep_nums):
#         start_i = ep_i * max_episode_steps
#         ep_returns[ep_i] = buffer.rewards[start_i:start_i+max_episode_steps, 0].sum()
#     return_mean = float(ep_returns.mean())
#     return_std = float(ep_returns.std())
#     return return_mean, return_std
#
#
# def load_model_sample_buffer(work_dir, load_step, buffer_size):
#     import json
#     from train import make_agent
#     from utils import FrameStack, eval_mode
#     from collections import namedtuple
#     import dmc2gym
#
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     print("device: " + str(device))
#
#     with open(f"{work_dir}/args.json") as fp:
#         args_dict = json.load(fp)
#         # 使用namedtuple进行字典转对象
#         Args = namedtuple("Args", args_dict.keys())
#         args = Args(**args_dict)
#
#     env = dmc2gym.make(
#         domain_name=args.domain_name,
#         task_name=args.task_name,
#         seed=args.seed,
#         visualize_reward=False,
#         from_pixels=(args.encoder_type == 'pixel'),
#         height=args.pre_transform_image_size,
#         width=args.pre_transform_image_size,
#         frame_skip=args.action_repeat
#     )
#
#     env.seed(args.seed)
#
#     if args.encoder_type == 'pixel':
#         env = FrameStack(env, k=args.frame_stack)
#
#     action_shape = env.action_space.shape
#
#     if args.encoder_type == 'pixel':
#         obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
#         pre_aug_obs_shape = (
#             3*args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
#     else:
#         obs_shape = env.observation_space.shape
#         pre_aug_obs_shape = obs_shape
#
#     agent = make_agent(obs_shape, action_shape, args, device)
#
#     agent.load(f"{work_dir}/model", load_step)
#
#     replay_buffer = ReplayBuffer(
#         obs_shape=pre_aug_obs_shape,
#         action_shape=action_shape,
#         capacity=buffer_size + 4,
#         batch_size=args.batch_size,
#         device=device,
#         image_size=args.image_size,
#     )
#
#     # start sample
#     with torch.no_grad():
#         done = True
#         episode_reward = 0
#         episode_step = 0
#         episode = 0
#         for i in range(buffer_size):
#             if done:
#                 obs = env.reset()  # 原来状态的维度为 (9, 100, 100)
#                 done = False
#                 episode_reward = 0
#                 episode_step = 0
#                 episode += 1
#
#             with eval_mode(agent):
#                 action = agent.sample_action(obs)
#             next_obs, reward, done, _ = env.step(action)
#
#             # allow infinit bootstrap
#             done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
#                 done
#             )
#             episode_reward += reward
#             replay_buffer.add(obs, action, reward, next_obs, done_bool)
#
#             obs = next_obs
#             episode_step += 1
#
#         mean, std = get_buffer_return_mean_std(replay_buffer, env._max_episode_steps)
#         buffer_save_path = f"{work_dir}/buffer/R[{int(mean)}&{int(std)}]S[{buffer_size}].pt"
#         replay_buffer.save_last_size(
#             buffer_size, save_path=buffer_save_path)
#
# if __name__ == "__main__":
#     # xvfb-run -a -s "-screen 0 1400x900x24" bash
#     work_dir = "tmp/cartpole/cartpole-swingup-01-01-23-13-im84-b64-s829529-pixel"
#     load_step = 200000
#     sample_size = 2000
#     load_model_sample_buffer(work_dir, load_step, sample_size)
