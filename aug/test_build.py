import torch
import torch_utils
from torch_utils.ops import bias_act, upfirdn2d

x = torch.randn(2, 3, 16, 16, device="cuda")
y = bias_act.bias_act(x, act="relu")
assert y.shape == x.shape

w = torch.randn(3, 3, device="cuda")
z = upfirdn2d.upfirdn2d(x, w, up=1, down=1)
assert z.shape[:2] == x.shape[:2] and z.ndim == x.ndim

print("bias_act impl:", "cuda" if bias_act._init() else "ref")
print("upfirdn2d impl:", "cuda" if upfirdn2d._init() else "ref")
print("OK")
