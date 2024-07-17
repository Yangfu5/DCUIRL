import numpy as np
import torch
import argparse
import os
import time
import json
import dmc2gym
from termcolor import colored

import utils
from logger import Logger
from video import VideoRecorder
from cuirl import CUIRL

from curl_sac import CurlSacAgent
from ctmr_sac import CtmrSacAgent

from utils import ReplayBuffer, load_expert_buffer, get_buffer_return_mean_std

expert_buffer_dict = {
    'ctmr_sac-cartpole-swingup': 'expert_buffer/cartpole/R[816&15]S[2000].pt',
    'ctmr_sac-cheetah-run': 'expert_buffer/cheetah/R[288&16]S[2000].pt',
    'ctmr_sac-finger-spin': 'expert_buffer/finger/R[455&27]S[2000].pt',
    # 'ctmr_sac-finger-spin': 'expert_buffer/finger/finger-spin.pt',
    'ctmr_sac-walker-walk': 'expert_buffer/walker/R[838&56]S[2000].pt'
}


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=50000, type=int)
    parser.add_argument('--buffer_save_capacity', default=2000, type=int)
    parser.add_argument('--expert_buffer_size', default=2000, type=int)
    # train
    parser.add_argument('--agent', default='ctmr_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int)  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_attn_layer', default=1, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='results', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--actor_attach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--exp_suffix', default='', type=str)
    parser.add_argument('--actor_coeff', default=1., type=float)
    parser.add_argument('--adam_warmup_step', type=float)
    parser.add_argument('--encoder_annealling', default=False, action='store_true')
    parser.add_argument('--mtm_length', default=20, type=int)
    parser.add_argument('--mtm_bsz', default=64, type=int)
    parser.add_argument('--mtm_not_ema', default=False, action='store_true')
    parser.add_argument('--mtm_ratio', default=0.25, type=float)
    parser.add_argument('--normalize_before', default=False, action='store_true')
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--attention_dropout', default=0., type=float)
    parser.add_argument('--relu_dropout', default=0., type=float)

    # IRL
    parser.add_argument('--is_irl', default=False, action='store_true')
    parser.add_argument('--is_not_update_cpc', default=False, action='store_true')
    parser.add_argument('--irl_per_updates', default=1, type=int)
    parser.add_argument('--irl_lr', default=1e-3, type=float)
    parser.add_argument('--irl_type', default='DAC', type=str)

    # RL
    parser.add_argument('--rl_per_updates', default=1, type=int)

    args = parser.parse_args()
    # update_args(args)
    if args.num_train_steps < 500000:
        args.num_train_steps = 500000
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    elif args.agent == 'ctmr_sac':
        return CtmrSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            num_attn_layer=args.num_attn_layer,
            actor_attach_encoder=args.actor_attach_encoder,
            actor_coeff=args.actor_coeff,
            adam_warmup_step=args.adam_warmup_step,
            encoder_annealling=args.encoder_annealling,
            mtm_bsz=args.mtm_bsz,
            mtm_ema=not args.mtm_not_ema,
            normalize_before=args.normalize_before,
            relu_dropout=args.relu_dropout,
            attention_dropout=args.attention_dropout,
            dropout=args.dropout,
        )

    else:
        assert 'agent is not supported: %s' % args.agent


def prepare():
    args = parse_args()
    print(args.batch_size)
    args.batch_size=64
    print(args.batch_size)
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
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

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H-%M", ts)
    exp_suffix = '-' + args.exp_suffix if args.exp_suffix else ''
    env_name = args.agent + '-' + args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) + '-b' \
               + str(args.batch_size) + '-' + args.encoder_type \
               + exp_suffix + '-s' + str(args.seed)
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print(json.dumps(vars(args), separators=(',', ':\t'), sort_keys=True, indent=4))  # 打印参数
    device = torch.device('cuda:{}'.format(args.cuda_id) if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        mtm_bsz=args.mtm_bsz,
        mtm_length=args.mtm_length,
        mtm_ratio=args.mtm_ratio
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if args.is_irl:
        irl = CUIRL(curl_sac=agent, action_dim=action_shape[0],
                    device=device, hidden_dim=args.hidden_dim,
                    lr=args.irl_lr, irl_type=args.irl_type, detach_encoder=True)
        expert_buffer_path = expert_buffer_dict[env_name]
        expert_buffer = load_expert_buffer(expert_buffer_path,
                                           buffer_size=args.expert_buffer_size,
                                           batch_size=args.batch_size,
                                           device=device,
                                           image_size=args.image_size
                                           )
        expert_return_mean, expert_return_std = get_buffer_return_mean_std(expert_buffer, env._max_episode_steps)
        print('*' * 30)
        print(expert_return_mean)
        print(expert_return_std)
        args.expert_return_mean = expert_return_mean
        args.expert_return_std = expert_return_std
    else:
        irl = None
        expert_buffer = None

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    return args, video, L, env, agent, irl, model_dir, replay_buffer, expert_buffer, buffer_dir


def run(args, video, L, env, agent: CtmrSacAgent, irl: CUIRL, model_dir, replay_buffer: ReplayBuffer,
        expert_buffer: ReplayBuffer, buffer_dir):
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    print('{}: {}'.format(colored('env._max_episode_steps', 'red'),
                          env._max_episode_steps))
    print('{}: {}'.format(colored('eval_freq', 'red'), args.eval_freq))
    assert args.eval_freq % env._max_episode_steps == 0
    for step in range(args.num_train_steps + 1):
        if step % 100 == 0:
            print(step)
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step, args)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # 更新模块
        is_not_update_cpc = args.is_not_update_cpc if hasattr(args, 'is_not_update_cpc') else False
        if step >= args.init_steps:
            # 更新逆强化学习模块
            if irl is not None:
                for _ in range(args.irl_per_updates):
                    irl.update(replay_buffer, expert_buffer, is_not_update_cpc, L, step)
            for _ in range(args.rl_per_updates):
                agent.update(replay_buffer, is_not_update_cpc, L, step, irl)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


def main():
    args, video, L, env, agent, irl, model_dir, replay_buffer, expert_buffer, buffer_dir = prepare()
    run(args, video, L, env, agent, irl, model_dir, replay_buffer, expert_buffer, buffer_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
