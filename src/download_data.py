from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from urllib import request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH

DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def download_dataset(url: str = DATASET_URL, output_dir: str | Path | None = None) -> Path:
    """Download the UCI Bank Marketing ZIP archive and extract bank-full.csv into data/raw/."""
    raw_dir = Path(output_dir) if output_dir is not None else RAW_DATA_PATH.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists():
        print(f"Data already exists at '{RAW_DATA_PATH}'. Skipping download.")
        return RAW_DATA_PATH

    temp_zip = raw_dir / "bank_marketing.zip"
    temp_extract_dir = raw_dir / "__temp_bank_extract__"

    try:
        print(f"Downloading dataset from {url}...")
        request.urlretrieve(url, temp_zip)

        with zipfile.ZipFile(temp_zip, "r") as archive:
            nested_archive_name = next(
                (name for name in archive.namelist() if name.endswith("bank.zip")),
                None,
            )
            if nested_archive_name is None:
                raise FileNotFoundError("The downloaded ZIP does not contain the expected nested 'bank.zip' archive.")

            archive.extract(nested_archive_name, temp_extract_dir)

        nested_zip_path = temp_extract_dir / nested_archive_name
        if not nested_zip_path.exists():
            raise FileNotFoundError(f"Nested archive not found at '{nested_zip_path}'.")

        with zipfile.ZipFile(nested_zip_path, "r") as inner_archive:
            csv_name = "bank-full.csv"
            if csv_name not in inner_archive.namelist():
                raise FileNotFoundError(f"The archive does not contain '{csv_name}'.")
            inner_archive.extract(csv_name, raw_dir)

        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"The expected raw file '{RAW_DATA_PATH}' was not created after extraction.")

        print(f"Dataset downloaded successfully to '{RAW_DATA_PATH}'.")
        return RAW_DATA_PATH

    except Exception as exc:  # pragma: no cover - runtime user guidance path
        raise RuntimeError(
            "Veri indirilemedi. Lütfen şu adresten ZIP dosyasını indirin ve içindeki 'bank-full.csv' dosyasını "
            "'data/raw/' klasörüne manuel olarak yerleştirin: "
            "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
        ) from exc

    finally:
        if temp_zip.exists():
            temp_zip.unlink()
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)


if __name__ == "__main__":
    download_dataset()
