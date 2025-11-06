from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download(url: str, target_path: str) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(target_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading ml-100k") as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    out_dir = data_dir / "ml-100k"
    if out_dir.exists():
        print(f"Exists: {out_dir}")
        return
    print("Downloading MovieLens 100K ...")
    download(URL, str(zip_path))
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(data_dir))
    print("Done.")


if __name__ == "__main__":
    main()

