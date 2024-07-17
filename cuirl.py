import torch.nn as nn
from curl_sac import CurlSacAgent, CURL, weight_init
from ctmr_sac import CtmrSacAgent, CTMR, weight_init
from encoder import PixelEncoder
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from utils import sigmoid_cross_entropy_with_logits
import torch.autograd as autograd


class CUIRL:
    # 需要传入cpc
    def __init__(self, curl_sac: CtmrSacAgent, action_dim, device, hidden_dim=32, lr=1e-3, irl_type="GAIL", detach_encoder=True) -> None:
        # 导入专家数据
        # 设置鉴别器
        self.curl_sac = curl_sac
        self.feature_dim = self.curl_sac.CTMR.encoder.feature_dim
        self.irl_type = irl_type  # "GAIL" or "DAC"
        self.lr = lr
        self.detach_encoder = True
        self.device = device
        self.lambd = 10

        if self.irl_type == "GAIL":
            self.DNet = nn.Sequential(
                nn.Linear(self.feature_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1), nn.Sigmoid()
            ).to(self.device)
        else:
            self.DNet = nn.Sequential(
                nn.Linear(self.feature_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.DNet.parameters(), lr=self.lr)
        weight_init(self.DNet)

    def encode(self, obs):
        new_obs = self.curl_sac.CTMR.encode(obs, detach=self.detach_encoder)
        return new_obs

    def update_by_GAIL(self, agent_sa, expert_sa, logger):
        error = 1e-5
        gene_D = self.DNet(agent_sa)
        expert_D = self.DNet(expert_sa)

        self.optimizer.zero_grad()
        """loss = log_pi_(D) + log_e_(1-D)"""
        loss = (torch.log(gene_D + error)).mean() + torch.log(1 - (expert_D - error)).mean()
        clip_grad_norm_(self.DNet.parameters(), 50.)
        loss.backward()
        self.optimizer.step()

    def update_by_DAC(self, gene_sa, expert_sa, logger):
        alpha = np.random.uniform(size=(gene_sa.shape[0], 1))
        alpha = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)
        inter_input = (alpha * gene_sa + (1. - alpha) * expert_sa).requires_grad_()  # 开启梯度

        gene_output = self.DNet(gene_sa)
        expert_output = self.DNet(expert_sa)
        inter_output = self.DNet(inter_input)
        gene_loss = sigmoid_cross_entropy_with_logits(gene_output, 0.)
        expert_loss = sigmoid_cross_entropy_with_logits(expert_output, 1.)

        # ? 为了计算二阶导数，这里一阶导数部分必须设置create_graph=True, retain_graph=True
        grad = autograd.grad(inter_output.sum(), inter_input, create_graph=True, retain_graph=True)[0]
        grad_penalty = torch.pow(torch.norm(grad, dim=-1) - 1, 2).mean()

        self.optimizer.zero_grad()
        loss = gene_loss + expert_loss + self.lambd * grad_penalty
        loss.backward()
        clip_grad_norm_(self.DNet.parameters(), 50.)
        self.optimizer.step()

    # 更新鉴别器
    def update(self, agent_buffer, expert_buffer, is_not_update_cpc, logger, step):
        # buffer sample出的数据已经保证在GPU上了
        obs, action, reward, next_obs, not_done, cpc_kwargs = agent_buffer.sample_curl()
        e_obs, e_action, e_reward, e_next_obs, e_not_done, e_cpc_kwargs = expert_buffer.sample_curl()
        obs = self.encode(obs)
        e_obs = self.encode(e_obs)
        agent_sa = torch.concat((obs, action), -1)
        expert_sa = torch.concat((e_obs, e_action), -1)

        if self.irl_type == "GAIL":
            self.update_by_GAIL(agent_sa, expert_sa, logger)
        else:  # "DAC"
            self.update_by_DAC(agent_sa, expert_sa, logger)

        if not is_not_update_cpc:
            self.curl_sac.update_cpc_irl(cpc_kwargs, logger, step)
            self.curl_sac.update_cpc_irl(e_cpc_kwargs, logger, step)

    # 获得奖励值
    def get_reward_s_a(self, obs, act):
        # 在IRL中不对encoder进行更新
        obs = self.encode(obs)
        sa = torch.concat((obs, act), dim=-1)
        reward = self.DNet(sa)
        return reward

    def get_reward_sa(self, sa):
        reward = self.DNet(sa)
        return reward
