import requests
import os

ARTICLE_ID = 22802147
OUT_DIR = "y1000plus_data"
os.makedirs(OUT_DIR, exist_ok=True)

# Get file list from public API
files = requests.get(
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"
).json()

for f in files:
    print(f["name"], f"({f['size']:,} bytes)")
    r = requests.get(f["download_url"], stream=True)
    path = os.path.join(OUT_DIR, f["name"])
    with open(path, "wb") as out:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            out.write(chunk)
    print(f"  → saved to {path}")
