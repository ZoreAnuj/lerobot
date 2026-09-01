"""fp32 vs bf16 vs bf16+compile for IMLE, batch 64 (what the training loop actually does)."""
import contextlib, time, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

CKPT = "/mnt/data/zero/runs/imle_dice_j7_2cam/checkpoints/100000/pretrained_model"
B, ITERS = 64, 14
torch.backends.cudnn.benchmark = True

def run(label, amp_dtype, compile_unet, tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.device = "cuda"
    cfg.compile_model = compile_unet
    cfg.compile_mode = "default"
    p = get_policy_class(cfg.type)(cfg).to("cuda"); p.train()
    opt = torch.optim.AdamW(p.parameters(), lr=1e-4)
    batch = {"observation.state": torch.randn(B, cfg.n_obs_steps, 7, device="cuda"),
             "action": torch.randn(B, cfg.horizon, 7, device="cuda"),
             "action_is_pad": torch.zeros(B, cfg.horizon, dtype=torch.bool, device="cuda")}
    for k in cfg.image_features:
        batch[k] = torch.rand(B, cfg.n_obs_steps, 3, 480, 640, device="cuda")
    ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
    for i in range(ITERS):
        if i == 4:
            torch.cuda.synchronize(); t0 = time.time()
        with ctx():
            loss, _ = p.forward(batch)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / (ITERS - 4)
    print(f"{label:34s} {B/dt:8.1f} samples/s  {dt*1000:7.1f} ms/step  "
          f"{torch.cuda.max_memory_allocated()/1e9:5.1f} GB", flush=True)
    del p, opt, batch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

run("fp32 (what our runs used)", None, False, False)
run("fp32 + TF32", None, False, True)
run("bf16 autocast", torch.bfloat16, False, True)
run("bf16 + compile(unet)", torch.bfloat16, True, True)
