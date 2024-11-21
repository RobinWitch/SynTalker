import pynvml

def get_gpt_id():
    return "1"
    pynvml.nvmlInit()
    gpu_indices = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        perf_state = pynvml.nvmlDeviceGetPowerState(handle)
        #if perf_state == 8 and memory_info.used < 2000 * 1024 * 1024:
        if perf_state == 8:
            gpu_indices.append(i)
    assert len(gpu_indices) > 0, "There is no GPU with performance state P8 and low memory usage"
    pynvml.nvmlShutdown()
    print(f"usalbe gpu ids: {gpu_indices} , now we use {gpu_indices[-1]}")
    return str(gpu_indices[-1])
