from huggingface_hub import snapshot_download

repo_id = "robinwitch/SynTalker"
local_dir = "./ckpt"
allow_patterns = ["ckpt/*"]

snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)

local_dir = "./datasets"
allow_patterns = ["datasets/*"]
snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
