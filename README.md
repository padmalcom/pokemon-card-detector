# pokemon-card-detector
A simple python app that classifies pokemon cards in real-time. Pokemon data is loaded via tcgdexsdk (thanks for hosting!).

## Quick start
- call pip install -r requirements.txt
- call query_card_webcam.py to classify cards in real-time via your webcam (card must fill the rectangle in the middle)
- call query_card.py to classify a photo

### How to re-create indices (normally not required)
- Data is gathered via download_data.py. This script downloads images and json meta-data to a "data" directory
- Merge the data into a single json file via merge_json.py
- Call build_index.py to create numpy arrays with emeddings and labels
- Call create_faiss_index.py to create the FAISS index
- You can now delete the data folder

## Install on raspberry pi
- sudo apt install python3-picamera2
- python3 -m venv pokemon --system-site-packages
- source pokemon/bin/activate
- pip install git+https://github.com/openai/CLIP.git
