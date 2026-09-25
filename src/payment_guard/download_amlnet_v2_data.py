"""Download and verify the AMLNet Version 2.0 transaction dataset."""

from argparse import ArgumentParser
from hashlib import new as new_hash
from pathlib import Path
from shutil import which
from subprocess import run


ZENODO_RECORD_ID = "21237971"
RAW_FILE_NAME = "AMLNet_v2_transactions.csv"
DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"
    f"{RAW_FILE_NAME}?download=1"
)
EXPECTED_MD5 = "0d3702e54495385bb8f01a688466a4e3"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "amlnet_v2"

CHUNK_SIZE = 8 * 1024 * 1024


def calculate_checksum(file_path: Path, algorithm: str) -> str:
    """Calculate a checksum without loading the complete file into memory."""
    checksum = new_hash(algorithm)

    with file_path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            checksum.update(chunk)

    return checksum.hexdigest()


def download_dataset(force: bool = False) -> Path:
    """Download the fixed dataset and verify its published checksum."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    destination = RAW_DATA_DIR / RAW_FILE_NAME
    temporary_file = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and not force:
        existing_md5 = calculate_checksum(destination, "md5")

        if existing_md5 == EXPECTED_MD5:
            existing_sha256 = calculate_checksum(destination, "sha256")
            print(f"SKIP {RAW_FILE_NAME}: existing file is valid")
            print(f"MD5 verified: {existing_md5}")
            print(f"SHA-256: {existing_sha256}")
            return destination

        raise RuntimeError(
            "The existing CSV does not match Zenodo's published MD5. "
            "Inspect it or run again with --force."
        )

    if which("curl") is None:
        raise RuntimeError("curl is required but was not found.")

    if force and temporary_file.exists():
        temporary_file.unlink()
        print(f"Removed temporary file: {temporary_file.name}")

    print(f"Downloading {RAW_FILE_NAME}")
    print(f"Source: {DOWNLOAD_URL}")

    if force:
        print("Force mode: starting a fresh temporary download.")
    else:
        print(
            "If a partial file exists, curl will attempt to resume it."
        )

    run(
        [
            "curl",
            "--fail",
            "--location",
            "--progress-bar",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(temporary_file),
            DOWNLOAD_URL,
        ],
        check=True,
    )

    downloaded_md5 = calculate_checksum(temporary_file, "md5")

    if downloaded_md5 != EXPECTED_MD5:
        raise RuntimeError(
            "Checksum verification failed. "
            f"Expected {EXPECTED_MD5}, received {downloaded_md5}. "
            f"The temporary file was kept at {temporary_file}."
        )

    temporary_file.replace(destination)
    downloaded_sha256 = calculate_checksum(destination, "sha256")

    print("Download complete")
    print(f"MD5 verified: {downloaded_md5}")
    print(f"SHA-256: {downloaded_sha256}")

    return destination


def parse_arguments():
    """Read command-line options."""
    parser = ArgumentParser(
        description="Download and verify AMLNet Version 2.0."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "start a fresh download even if the destination file "
            "already exists"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset downloader."""
    arguments = parse_arguments()
    download_dataset(force=arguments.force)


if __name__ == "__main__":
    main()
