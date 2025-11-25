import json
import os
from pathlib import Path

def combine_json_files(input_dir: str, output_file: str):
    """Scan input_dir for .json files, combine them into one dict, and save to output_file."""
    combined = {}
    input_path = Path(input_dir)
    json_files = list(input_path.rglob("*.json"))  # recursively find all .json
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load JSON from {file_path}: {e}")
            continue
        
        # Assume each JSON is a dict with an "id" field
        if not isinstance(data, dict):
            print(f"WARNING: JSON root is not an object in {file_path}, skipping")
            continue
        
        if "id" not in data:
            print(f"WARNING: No 'id' key in {file_path}, skipping")
            continue
        
        entry_id = data["id"]
        
        # Copy everything except 'id'
        # (Alternatively, you could keep the entire object including id)
        entry_value = {k: v for k, v in data.items() if k != "id"}
        
        if entry_id in combined:
            print(f"WARNING: Duplicate id {entry_id} from file {file_path}. Overwriting previous.")
        
        combined[entry_id] = entry_value
    
    # Save the combined dict to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    print(f"Saved combined JSON with {len(combined)} entries to {output_file}")


if __name__ == "__main__":
    input_directory = "data"
    output_json = "cards.json"
    combine_json_files(input_directory, output_json)
