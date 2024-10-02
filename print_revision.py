#!/usr/bin/env python

from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info("black-forest-labs/FLUX.1-schnell")
latest_revision = info.sha
print(f"Latest revision: FLUX.1-schnell - {latest_revision}")

info = api.repo_info("black-forest-labs/FLUX.1-dev")
latest_revision = info.sha
print(f"Latest revision: FLUX.1-dev     - {latest_revision}")
