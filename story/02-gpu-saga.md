# 02 · GPU Saga

*The most technically exhausting part of this session had nothing to
do with machine learning. It had to do with keeping an H100 alive
long enough to train on it.*

## Act I: `cuInit = 802`

The first H100-equipped VM (`ubuntu@217.18.55.122`) had PyTorch 2.6
installed, `nvidia-smi` reported the H100 cleanly, and `uname -a`
showed a sensible Ubuntu 22.04 kernel. But a trivial probe failed:

```python
>>> import torch
>>> torch.cuda.is_available()
/.../torch/cuda/__init__.py:129: UserWarning:
CUDA initialization: Unexpected error from cudaGetDeviceCount().
Error 802: system not yet initialized
False
```

Error 802 is `CUDA_ERROR_SYSTEM_NOT_READY`. The device is present,
the kernel module is loaded, the user-mode `libcuda.so.1` resolves
correctly — but something downstream of `cuInit()` has not
initialised. This error routinely shows up on H100 SXM nodes that
need the fabricmanager service; it is much less common on PCIe
single-card systems, where fabricmanager should be a no-op.

## Act II: fabricmanager does not help

Five hypotheses were tested in sequence:

1. **Stale `LD_LIBRARY_PATH`** pointing at an incompatible CUDA 13.0
   userland while torch was built against cu124. Unset — no change.
2. **Driver module reload** (`rmmod nvidia_uvm nvidia` then
   `modprobe nvidia` then `modprobe nvidia_uvm`) — temperature
   readings came back and persistence-mode worked, but `cuInit`
   still returned 802.
3. **GPU reset** via `nvidia-smi --gpu-reset -i 0` — rejected with
   `Not Supported`, which is typical under KVM-passthrough where
   function-level reset is trapped by the hypervisor.
4. **fabricmanager install** — the apt repo offered
   `nvidia-fabricmanager-580.126.09`, but the driver was
   `580.126.20`. The version mismatch refused to start the service.
5. **Fabricmanager 580.126.20 from NVIDIA's CUDA redist** — the new
   binary started, but logged `NV_WARN_NOTHING_TO_DO` because there
   is no NVSwitch on a single-card PCIe system. Expected and
   harmless, but it did not fix 802 either.

## Act III: root cause

`dmesg` told the real story:

```
NVRM: gpuHandleSanityCheckRegReadError_GH100:
      Possible bad register read: addr: 0x110040, regvalue: 0xbadf4100
NVRM: _threadNodeCheckTimeout: Timeout was set to: 4000 msecs!
NVRM: kflcnWaitForHaltRiscv_GA102:
      Timeout waiting for RISC-V to halt
NVRM: nvAssertFailedNoLog:
      Assertion failed: rmStatus == NV_OK @ osinit.c:2350
```

The `0xbadf4100` sentinel is NVIDIA's "bad register read" magic; the
RISC-V halt timeout is the GSP firmware failing to initialise.
Combined with `Virtualization Mode: Pass-Through` and
`GPU Recovery Action: Reset`, this is a GPU that dropped off the PCIe
bus during a previous tenant's workload. In a passthrough VM, the
guest cannot issue the PCIe reset that would recover it; the
hypervisor has to.

## Act IV: migration

Rather than spend more hours arguing with a permanently-wedged GPU,
we requested a fresh VM (`ubuntu@217.18.55.93`). Within two minutes:

- `cuInit()` returned `CUDA_SUCCESS`.
- `torch.cuda.is_available()` returned `True`.
- `arch_list` reported every compute capability through `sm_90`.
- An FP16 `matmul(8192, 8192)` ran at **639.7 TFLOPS** — ~ 65 % of
  the H100's ~ 989 TFLOPS theoretical peak, which is healthy with
  kernel-launch overhead.

The dead H100's artefacts are preserved in this story for the same
reason commit logs preserve failed approaches: the **decision** to
migrate is only defensible if you can see what was tried first.

## What we learned

- **Check `dmesg` before trusting `nvidia-smi`.** `nvidia-smi` can
  report a GPU that the kernel driver has already given up on;
  `dmesg` is the source of truth.
- **`CUDA_ERROR_SYSTEM_NOT_READY` has many causes** — fabricmanager
  is one, GSP firmware halt is another, cgroup isolation is a
  third. The error code itself is not diagnostic.
- **On a passthrough VM, if `--gpu-reset` says `Not Supported`,
  migrate.** Everything short of VM restart is either a no-op or a
  distraction.
- **Keep artefacts for failed attempts.** Every command in this
  document has a timestamp and an `nvidia-smi` snapshot associated
  with it in the raw transcript.
