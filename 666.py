import torch
from torch.nn import MultiheadAttention
from torch.utils.benchmark import Timer
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义注意力层参数
embed_dim = 512  # 嵌入维度
num_heads = 512    # 注意力头数
seq_length = 64  # 序列长度
batch_size = 32  # 批量大小

# 创建输入张量
query = torch.rand(batch_size, seq_length, embed_dim).to(device)
key = torch.rand(batch_size, seq_length, embed_dim).to(device)
value = torch.rand(batch_size, seq_length, embed_dim).to(device)

# 单头注意力
single_head_attn = MultiheadAttention(embed_dim, 1).to(device)

# 多头注意力
multi_head_attn = MultiheadAttention(embed_dim, num_heads).to(device)

# 定义计时器
timer = Timer(
    stmt='attn(query, key, value)',
    setup='from __main__ import single_head_attn, multi_head_attn, query, key, value',
    globals={'attn': single_head_attn, 'query': query, 'key': key, 'value': value},
)

# 测量单头注意力的性能
single_head_time = timer.timeit(100)
single_head_flops = FlopCountAnalysis(single_head_attn, (query, key, value))

# 测量多头注意力的性能
timer._globals['attn'] = multi_head_attn
multi_head_time = timer.timeit(100)
multi_head_flops = FlopCountAnalysis(multi_head_attn, (query, key, value))

# 输出结果
print(f"Single-head Attention Time: {single_head_time.mean} seconds")
print(f"Single-head Attention FLOPs: {single_head_flops.total()} FLOPs")
print(f"Single-head Attention Parameters: {parameter_count_table(single_head_attn)}")

print(f"Multi-head Attention Time: {multi_head_time.mean} seconds")
print(f"Multi-head Attention FLOPs: {multi_head_flops.total()} FLOPs")
print(f"Multi-head Attention Parameters: {parameter_count_table(multi_head_attn)}")
