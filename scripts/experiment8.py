"""This script is used to test the HF API."""

from senselab.utils.hf import _check_hf_repo_exists

res = _check_hf_repo_exists("gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "model")
print(res)
