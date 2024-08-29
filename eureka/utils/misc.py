import subprocess
import os
import json
import logging

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    try:
        # Run nvidia-smi command to get GPU information in JSON format
        sp = subprocess.Popen(['nvidia-smi', '--query-gpu=index,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str, err_str = sp.communicate()

        # Check if there was an error
        if sp.returncode != 0:
            logging.error(f"Error executing nvidia-smi: {err_str.decode('utf-8', errors='ignore')}")
            return 0  # Return a default GPU index or handle it as needed

        # Parse the output
        gpu_info = out_str.decode('utf-8').strip().split('\n')
        gpu_stats = []
        for line in gpu_info:
            index, memory_used, memory_free = map(int, line.split(','))
            gpu_stats.append({'index': index, 'memory.used': memory_used, 'memory.free': memory_free})

        # Find GPU with most free memory
        freest_gpu = max(gpu_stats, key=lambda x: x['memory.free'])
        return freest_gpu['index']

    except Exception as e:
        logging.error(f"An error occurred while getting GPU info: {e}")
        return 0  # Return a default GPU index or handle it as needed

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break

if __name__ == "__main__":
    print(get_freest_gpu())