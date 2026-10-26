import os
import json
from sc2_datasets.torch.sc2_egset_dataset import SC2EGSetDataset
from sc2_datasets.available_replaypacks import EXAMPLE_SYNTHETIC_REPLAYPACKS

DOWNLOAD_DIR = "./data/downloads"
UNPACK_DIR = "./data/unpacked"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(UNPACK_DIR, exist_ok=True)

print("--> Ingesting SC2EGSet sample files...")

# Load dataset using names_urls parameter
dataset = SC2EGSetDataset(
    download_dir=DOWNLOAD_DIR,
    unpack_dir=UNPACK_DIR,
    download=True,
    names_urls=EXAMPLE_SYNTHETIC_REPLAYPACKS
)

print(f"\n--> Successfully verified dataset download ({len(dataset)} record(s)).\n")

# Recursive file inspector
all_files = []
for root, dirs, files in os.walk(UNPACK_DIR):
    for f in files:
        all_files.append(os.path.join(root, f))

print(f"================ DISCOVERED DATA FILES ({len(all_files)}) ================")
for filepath in all_files[:10]:
    rel_path = os.path.relpath(filepath, UNPACK_DIR)
    file_size = os.path.getsize(filepath) / 1024  # Size in KB
    print(f" - Path: {rel_path} | Size: {file_size:.2f} KB")

print("=================================================================\n")

# Read a sample file if found
json_files = [f for f in all_files if f.endswith('.json')]
if json_files:
    sample_path = json_files[0]
    print(f"--> Reading sample match log: {os.path.basename(sample_path)}")
    with open(sample_path, 'r', encoding='utf-8') as f:
        game_data = json.load(f)
    
    print("\n--- SAMPLE MATCH METADATA ---")
    print(f" Map Name : {game_data.get('map_name', game_data.get('Map', 'N/A'))}")
    print(f" Duration : {game_data.get('game_duration_s', game_data.get('Duration', 'N/A'))}")
    print(f" Matchup  : {game_data.get('matchup', game_data.get('Matchup', 'N/A'))}")
    print("------------------------------\n")

print("Step 1 Complete! Data ingestion is confirmed.")