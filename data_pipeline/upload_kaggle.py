"""
Automated Kaggle dataset packaging and upload script.
Generates dataset-metadata.json and utilizes the Kaggle API to upload and publish the dataset.
"""

from typing import Optional
import os
import sys
import json
import subprocess


class KaggleDatasetUploader:
    """
    Packages dataset directory, writes metadata, and uploads to Kaggle via Kaggle API.
    """
    def __init__(
        self,
        dataset_dir: str,
        title: str = "Sentinel-2 TSViT Crop Classification (24x24)",
        dataset_slug: str = "sentinel2-tsvit-crop-classification-24x24",
        username: Optional[str] = None,
        license_name: str = "CC0-1.0"
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.title = title
        self.dataset_slug = dataset_slug
        self.license_name = license_name
        self.username = username or os.environ.get("KAGGLE_USERNAME")

    def create_metadata(self) -> str:
        """Writes dataset-metadata.json into dataset directory."""
        if not self.username:
            # Fallback to reading from ~/.kaggle/kaggle.json
            kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
            if os.path.exists(kaggle_json):
                with open(kaggle_json, "r") as f:
                    data = json.load(f)
                    self.username = data.get("username", "user")
            else:
                self.username = "user"

        metadata = {
            "title": self.title,
            "id": f"{self.username}/{self.dataset_slug}",
            "licenses": [{"name": self.license_name}],
            "description": "Multi-temporal Sentinel-2 10-band satellite imagery patches (24x24) with paired RPG 2022 agricultural crop classification labels."
        }

        metadata_path = os.path.join(self.dataset_dir, "dataset-metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Kaggle] Created dataset-metadata.json at: {metadata_path}")
        return metadata_path

    def upload_dataset(self, make_public: bool = True) -> bool:
        """
        Uploads or versions the dataset using Kaggle CLI.
        """
        self.create_metadata()

        print(f"[Kaggle] Initiating upload from directory: {self.dataset_dir}")
        cmd = ["kaggle", "datasets", "create", "-p", self.dataset_dir]
        if make_public:
            cmd.append("--public")

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                print(f"[Kaggle] Successfully uploaded dataset:\n{res.stdout}")
                return True
            else:
                # If dataset already exists, try creating a new version
                if "already exists" in res.stderr or "already exists" in res.stdout:
                    print("[Kaggle] Dataset already exists, creating new version...")
                    ver_cmd = ["kaggle", "datasets", "version", "-p", self.dataset_dir, "-m", "Updated dataset patches"]
                    ver_res = subprocess.run(ver_cmd, capture_output=True, text=True, check=False)
                    print(ver_res.stdout)
                    return ver_res.returncode == 0
                else:
                    print(f"[Kaggle] Upload warning: {res.stderr}\nMake sure Kaggle API credentials (~/.kaggle/kaggle.json) are configured.")
                    return False
        except FileNotFoundError:
            print("[Kaggle] Kaggle CLI is not installed or not in PATH. Install via `pip install kaggle`.")
            return False


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    uploader = KaggleDatasetUploader(dataset_dir=target_dir)
    uploader.create_metadata()
