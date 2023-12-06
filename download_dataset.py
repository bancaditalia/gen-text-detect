import os
import requests
from tqdm import tqdm


def get_remote_file(filename, remote_prefix, new_filename=None):
    if new_filename is None:
        new_filename = filename
    if os.path.exists(os.path.join(subdir, new_filename)):
        print(f"File '{new_filename}' already exists. It will not be downloaded again.")
    else:
        r = requests.get(remote_prefix + filename, stream=True)

        with open(os.path.join(subdir, new_filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


subdir = 'data'
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\','/') # needed for Windows

# download WebText-GPT2 dataset
for ds in [
    'webtext',
    'small-117M',  'small-117M-k40',
    'medium-345M', 'medium-345M-k40',
    'large-762M',  'large-762M-k40',
    'xl-1542M',    'xl-1542M-k40',
]:
    for split in ['train', 'valid', 'test']:
        filename = f"{ds}.{split}.jsonl"
        if ds == 'webtext':
            target_filename = filename
        else:
            target_filename = "GPT2-" + filename
        get_remote_file(f"{ds}.{split}.jsonl",
                        "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/",
                        target_filename)

# download Grover dataset
grover_size = ['base', 'medium', 'mega']
grover_p = ['0.90', '0.92', '0.94', '0.96', '0.98', '1.00']
for size in grover_size:
    for p in grover_p:
        get_remote_file(f"generator={size}~dataset=p{p}.jsonl",
                        "https://storage.googleapis.com/grover-models/generation_examples/",
                        f"Grover-{size}-p{p}.test.jsonl")

# download GPT-3 sample generations
get_remote_file("175b_samples.jsonl",
                "https://raw.githubusercontent.com/openai/gpt-3/master/",
                "GPT3-175b.test.jsonl")