from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import zipfile
import boto3
import json
import sys
import os

COMPETITION = "understanding_cloud_organization"
S3_PREFIX = "raw/"

load_dotenv()

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ["S3_BUCKET_NAME"]


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_REGION,
    )


def _kaggle_env() -> dict:
    env = os.environ.copy()
    credentials_path = Path.home() / ".kaggle" / "credentials.json"
    if credentials_path.exists():
        creds = json.loads(credentials_path.read_text())
        if "access_token" in creds:
            env["KAGGLE_TOKEN"] = creds["access_token"]
    return env


def download_competition(dest: Path):
    cmd = [
        sys.executable, "-m", "kaggle",
        "competitions", "download",
        "-c", COMPETITION,
        "-p", str(dest),
    ]
    result = subprocess.run(cmd, env=_kaggle_env())
    if result.returncode != 0:
        raise RuntimeError(
            "kaggle download failed — check your credentials and that you "
            "accepted the competition rules at kaggle.com."
        )


def upload_file(s3, local_path: Path, s3_key: str):
    file_size = local_path.stat().st_size
    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=f"  -> s3://{S3_BUCKET}/{s3_key}",
        leave=False,
    ) as progress:
        s3.upload_file(
            str(local_path),
            S3_BUCKET,
            s3_key,
            Callback=lambda bytes_transferred: progress.update(bytes_transferred),
        )


def main():
    s3 = get_s3_client()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        print(f"Downloading competition files for '{COMPETITION}'...")
        download_competition(tmp)

        zip_files = list(tmp.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError("No zip files found after download.")

        for zip_path in zip_files:
            print(f"\nExtracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            zip_path.unlink()

        all_files = [f for f in tmp.rglob("*") if f.is_file()]
        print(f"\nUploading {len(all_files)} files to s3://{S3_BUCKET}/{S3_PREFIX}")

        for local_file in tqdm(all_files, desc="Overall progress"):
            relative = local_file.relative_to(tmp)
            s3_key = S3_PREFIX + str(relative)
            upload_file(s3, local_file, s3_key)

    print("\nDone. All files uploaded successfully.")


if __name__ == "__main__":
    main()
