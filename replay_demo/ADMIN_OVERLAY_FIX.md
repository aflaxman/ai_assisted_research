# Singularity Overlay Limit Issue - Admin Configuration Needed

## Current Status

✅ `ldconfig path` is now configured correctly in `singularity.conf`
❌ Still failing with "underlay required more than 50 bind mounts" warnings

## The Problem

When using `--nv` flag, Singularity tries to overlay host NVIDIA libraries onto the container's filesystem. The current configuration has a limit that's being exceeded:

```
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (409) bind mounts
WARNING: underlay of /etc/localtime required more than 50 (88) bind mounts
```

This prevents GPU access even though all libraries and devices are available.

## System Details

- **Singularity Version**: CE 4.1.4
- **Config Location**: `/opt/singularity/etc/singularity/singularity.conf`
- **NVIDIA Driver**: 535.216.03
- **GPUs**: 8x Tesla V100-SXM2-32GB

## Solution: Increase Overlay Limits

Edit `/opt/singularity/etc/singularity/singularity.conf` and add/modify these settings:

```conf
# Increase maximum number of binds for overlay filesystem
# Default is often 50, but NVIDIA --nv needs more
max loop devices = 256

# Allow more bind mounts
# (Check for similar settings like "max bind mounts" or "overlay limit")
```

### Finding the Right Setting

Search the config file for overlay-related settings:

```bash
grep -i "loop\|overlay\|bind.*max\|max.*bind" /opt/singularity/etc/singularity/singularity.conf
```

Common settings that might need adjustment:
- `max loop devices`
- `max overlay fs layers`
- `enable overlay`
- `shared loop devices`

## Alternative Workaround (if increasing limits doesn't work)

Modify `nvliblist.conf` to use explicit paths and reduce the number of files being bound:

Edit `/opt/singularity/etc/singularity/nvliblist.conf` to include ONLY essential libraries:

```
# Minimal library list
libcuda.so
libnvidia-ml.so
libnvidia-ptxjitcompiler.so
libnvidia-fatbinaryloader.so

# Devices
nvidia-smi
```

Remove all the GL/GLX/EGL libraries that aren't needed for compute-only workloads.

## Testing After Fix

```bash
# Should work without warnings
singularity exec --nv seir_cuda_minimal.sif python3 -c "from numba import cuda; print(cuda.is_available())"
```

Expected output: `True` with no overlay warnings.

## References

- Singularity GPU Support: https://docs.sylabs.io/guides/latest/user-guide/gpu.html
- GitHub Issue: https://github.com/sylabs/singularity/issues/various-overlay-issues
- Our diagnostic scripts: `check_overlay_conflict.sh`, `test_nv_variations.sh`

## Contact

- User: abie@gen-nvidia-gpu-d01
- Project: `~/projects/ai_assisted_research/replay_demo`
