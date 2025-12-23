import torch

# 加载模型
checkpoint = torch.load("models/blueprint.pt", map_location='cpu')
model = checkpoint['model']

print("=== 模型基本信息 ===")
print(f"模型类型: {type(model).__name__}")

# 获取参数键名
if hasattr(model, 'state_dict'):
    keys = list(model.state_dict().keys())
else:
    keys = list(model.keys()) if isinstance(model, dict) else []

print(f"总参数数: {len(keys)}")

print("\n=== 关键组件检测 ===")
# 策略相关
policy_keys = [k for k in keys if any(word in k.lower() for word in ['policy', 'action', 'logit'])]
print(f"策略相关: {len(policy_keys)}")
if policy_keys: print(f"  例如: {policy_keys[0]}")

# 价值相关
value_keys = [k for k in keys if any(word in k.lower() for word in ['value', 'critic'])]
print(f"价值相关: {len(value_keys)}")
if value_keys: print(f"  例如: {value_keys[0]}")

# 编码器相关  
encoder_keys = [k for k in keys if any(word in k.lower() for word in ['encoder', 'embed', 'transformer'])]
print(f"编码器相关: {len(encoder_keys)}")
if encoder_keys: print(f"  例如: {encoder_keys[0]}")

print("\n=== 模型主要方法 ===")
methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
key_methods = [m for m in methods if any(word in m for word in ['forward', 'predict', 'value', 'policy'])]
print("关键方法:", key_methods[:5])

print("\n=== 前10个参数名 ===")
for i, key in enumerate(keys[:10]):
    print(f"  {key}")