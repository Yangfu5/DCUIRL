import torch

file_path = 'expert_buffer/cartpole/R[816&15]S[2000].pt'

data = torch.load(file_path)
print(data)

# 查看 expert_data 的维度
# 重新获取专家数据
