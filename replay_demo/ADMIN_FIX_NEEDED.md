# Singularity GPU Access Issue - Admin Fix Required

## Problem Summary

Singularity's `--nv` flag fails to enable GPU access with the error:
```
WARNING: Could not find any nv libraries on this host!
CUDA_ERROR_NO_DEVICE (100)
```

This occurs because the `ldconfig path` is not configured in `singularity.conf`.

## System Details

- **Machine**: gen-nvidia-gpu-d01
- **GPUs**: 8x Tesla V100-SXM2-32GB (all detected by kernel)
- **NVIDIA Driver**: 535.216.03 (loaded and working)
- **Singularity**: CE 4.1.4 (installed in `/opt/singularity`)
- **NVIDIA Libraries**: Present in `/usr/lib/x86_64-linux-gnu/` (all 57+ libraries found)
- **Device Files**: All `/dev/nvidia*` devices exist and are accessible

## Root Cause

The file `/opt/singularity/etc/singularity/singularity.conf` is missing the `ldconfig path` configuration. When `--nv` flag is used, Singularity needs to run `ldconfig` to discover NVIDIA libraries, but it doesn't know where the `ldconfig` binary is located.

## The Fix (ONE LINE)

Edit `/opt/singularity/etc/singularity/singularity.conf` and add:

```conf
ldconfig path = /usr/sbin/ldconfig
```

This can go in the section with other path configurations. Search for commented-out `ldconfig` lines or add it near the top of the file.

## Verification After Fix

Users should be able to run:

```bash
singularity exec --nv <container>.sif python3 -c "from numba import cuda; print(cuda.is_available())"
```

And see `True` instead of `False`.

## Alternative: Update nvliblist.conf (if ldconfig is intentionally disabled)

If ldconfig is disabled for security reasons, you can manually specify library paths in:
`/opt/singularity/etc/singularity/nvliblist.conf`

Add full paths like:
```
/usr/lib/x86_64-linux-gnu/libcuda.so
/usr/lib/x86_64-linux-gnu/libnvidia-ml.so
...
```

## Impact

Currently, all GPU users on this system cannot use Singularity containers with NVIDIA GPUs. The `--nv` flag is non-functional for all users.

## Testing

We've created `diagnose_cuda_issue.sh` which will confirm the fix works:

```bash
./diagnose_cuda_issue.sh
```

This shows before/after comparison of host CUDA access vs container CUDA access.

## References

- Singularity NVIDIA GPU Support: https://sylabs.io/guides/latest/user-guide/gpu.html
- Our diagnostic scripts: `diagnose_cuda_issue.sh`, `test_gpu_permutations.sh`
- Original issue: Users cannot run GPU-accelerated SEIR simulations

## Contact

- User: abie@gen-nvidia-gpu-d01
- Project: `/home/abie/projects/ai_assisted_research/replay_demo`
