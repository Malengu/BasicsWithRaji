# This file is to test FSDP's initialization on a single GPU!

# import necessary libraries
import os
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

cmd = """
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 mini_bug_reproduce.py # raise error
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 mini_bug_reproduce.py # pass
"""

torch.distributed.init_process_group(backend='nccl')
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.set_default_device(local_rank)
device = torch.get_default_device()

class Test(nn.Module):

    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(5, 4)

    def forward(self, x):
        x = self.e(x)
        return x.sum()

model = Test().half()

model = FSDP(
    model,
    use_orig_params=True,
)

## copied from https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py#L1679-1719
upcasted_log = []
for module in FSDP.fsdp_modules(model):
    if not module._has_params:
        continue  # skip if FSDP module not managing parameters
    param = module._flat_param
    if (
        param.dtype != torch.float32
        and param.device != torch.device("meta")
        and param.requires_grad
    ):
        # keep log of names_params that was upcasted
        # NOTE: resorted to this because warnings.simplefilter("once") is somehow not working
        name_param_log = (module.module.__class__.__name__, ", ".join(module._flat_param._fqns))
        if name_param_log not in upcasted_log:
            upcasted_log.append(name_param_log)

        # this works because of FSDP's _runtime_utils.lazy_init.
        # Have to be careful not to call anything before this that
        # triggers lazy_init (e.g., _is_fsdp_root).
        param.data = param.data.to(torch.float32)  # upcasting
        module._handle._orig_param_dtype = torch.float32  # update

x = torch.randint(0, 5, (20,), device=device)

model.eval()
with torch.no_grad():
    loss = model(x)
