import subprocess
import time

from fla.utils import device_platform


def get_xpu_memory_usage():
    """Run `xpu-smi stats -d 0` and parse its output."""
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", "0"],
            capture_output=True,
            text=True,
            check=True,
        )
        return extract_gpu_memory_used(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run xpu-smi: {e}")
        return None


def extract_gpu_memory_used(output):
    """Extract 'GPU Memory Used (MiB)' from xpu-smi output."""
    # Use regex to find the line containing "GPU Memory Used (MiB)"
    output = output.strip().replace(" ", "")

    prefix = "|GPUMemoryUsed(MiB)|"
    start_idx = output.find(prefix)
    value_start = start_idx + len(prefix)
    end_idx = output.find("|", value_start+2)
    value_str = output[value_start:end_idx-2].strip()
    if value_str:
        return int(value_str)
    else:
        print("Could not find GPU memory usage in xpu-smi output.")
        return None


def get_nvgpu_memory_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    memory_used = int(result.stdout.strip())
    return memory_used


def check_gpu_memory():
    max_memory_mib = 4096  # Threshold in MiB (4 GB)
    max_wait_time = 3600    # 60 minutes in seconds
    sleep_time = 30       # Sleep for 30 seconds

    start_time = time.time()

    while True:

        # Extract GPU memory usage
        if device_platform == 'intel':
            # memory_used_mib = get_xpu_memory_usage()
            # since xpu-smi have conflicts in apt
            memory_used_mib = 0
        elif device_platform == 'nvidia':
            memory_used_mib = get_nvgpu_memory_usage()
        if memory_used_mib is None:
            exit(1)

        print(f"Current GPU memory usage: {memory_used_mib} MiB")

        if memory_used_mib > max_memory_mib:
            print(f"GPU memory usage exceeds {max_memory_mib} MiB. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            print("GPU memory usage is within limits.")
            exit(0)

        if time.time() - start_time > max_wait_time:
            print("GPU memory usage remains high for 10 minutes. Skipping this action.")
            exit(1)


if __name__ == "__main__":
    check_gpu_memory()
