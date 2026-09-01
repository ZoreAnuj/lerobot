"""Where does IMLE's compute go: the 4 ResNet18 encoder passes or the 20 U-Net candidates?"""
import time, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

CKPT="/mnt/data/zero/runs/imle_dice_j7_2cam/checkpoints/100000/pretrained_model"
B, IT = 64, 12
torch.backends.cudnn.benchmark = True
cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device="cuda"
p = get_policy_class(cfg.type)(cfg).to("cuda"); p.train()
m = p.imle
imgs = {k: torch.rand(B, cfg.n_obs_steps, 3, 480, 640, device="cuda") for k in cfg.image_features}
state = torch.randn(B, cfg.n_obs_steps, 7, device="cuda")
gcond = torch.randn(B, m.unet.global_cond_dim if hasattr(m.unet,'global_cond_dim') else 0, device="cuda")

def timeit(fn, label, dtype):
    for i in range(IT):
        if i == 3:
            torch.cuda.synchronize(); t0 = time.time()
        ctx = torch.autocast("cuda", dtype=dtype) if dtype else torch.enable_grad()
        with ctx:
            out = fn()
        (out.float().sum()).backward()
        p.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"  {label:28s} {(time.time()-t0)/(IT-3)*1000:7.1f} ms")

enc_keys = list(cfg.image_features)
def encoders():
    outs = []
    for i, k in enumerate(enc_keys):
        x = imgs[k].flatten(0, 1)                       # (B*n_obs, 3, 480, 640)
        enc = m.rgb_encoder[i] if isinstance(m.rgb_encoder, torch.nn.ModuleList) else m.rgb_encoder
        outs.append(enc(x))
    return torch.cat(outs, dim=-1)

with torch.no_grad():
    gc_dim = encoders().shape[-1] * cfg.n_obs_steps + 7 * cfg.n_obs_steps
cond = torch.randn(B, gc_dim, device="cuda")
noise = torch.randn(B * cfg.n_samples_per_condition, cfg.horizon, 7, device="cuda")
cond_rep = cond.repeat_interleave(cfg.n_samples_per_condition, 0)

for dtype, name in ((None, "fp32"), (torch.bfloat16, "bf16")):
    print(f"{name}:")
    timeit(encoders, f"{len(enc_keys)} ResNet18 encoders", dtype)
    timeit(lambda: m.unet(noise, global_cond=cond_rep), f"U-Net x{cfg.n_samples_per_condition} candidates", dtype)
