import subprocess


def get_gpu_generation() -> str | None:
    """Returns the GPU generation, if available."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    generation = result.stdout.strip().split("\n")

    if not generation:
        return None

    return ", ".join([info.strip() for info in generation])


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available on the system."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_gpu_docker_args() -> dict:
    """Get Docker arguments for GPU support if available.
    
    Returns:
        dict: Dictionary containing docker run arguments for GPU support
    """
    if has_nvidia_gpu():
        return {
            "gpus": "all",
            "environment": {
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"
            }
        }
    else:
        return {}
