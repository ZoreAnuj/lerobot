"""Push the rebuilt dataset to the Hub (private), with its dataset card."""
import os
import shutil

from huggingface_hub import HfApi

OUT = "/home/zero/matter/imle/datasets/dice_white_pnp_nodip"
CARD = "/tmp/claude-1000/-home-zero-matter-imle/1ee9319f-ee95-4f6f-8a2d-25bcbccae2d3/scratchpad/nodip_dataset_card.md"
REPO = "azorematter/dice_white_pnp_nodip"

if os.path.exists(f"{OUT}/.bounds.json"):
    os.remove(f"{OUT}/.bounds.json")
shutil.copy(CARD, f"{OUT}/README.md")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
print(f"repo ready: https://huggingface.co/datasets/{REPO}")

api.upload_large_folder(
    repo_id=REPO,
    folder_path=OUT,
    repo_type="dataset",
    num_workers=8,
    print_report=True,
    print_report_every=60,
)
files = api.list_repo_files(REPO, repo_type="dataset")
print(f"uploaded: {len(files)} files "
      f"({sum(1 for f in files if f.endswith('.mp4'))} mp4, "
      f"{sum(1 for f in files if f.endswith('.parquet'))} parquet)")
print(f"DONE https://huggingface.co/datasets/{REPO}")
