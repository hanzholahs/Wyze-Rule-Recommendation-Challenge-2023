with open(".access", "r") as f:
    api_token = f.read()
    
from huggingface_hub import snapshot_download
snapshot_download(repo_id="wyzelabs/RuleRecommendation",
                  repo_type="dataset",
                  local_dir="data/",
                  token = api_token)