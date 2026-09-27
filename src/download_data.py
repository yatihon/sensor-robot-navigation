from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DATASET_URL = ("https://archive.ics.uci.edu/static/public/194/wall%2Bfollowing%2Brobot%2Bnavigation%2Bdata.zip")
REQUIRED_FILES = ("sensor_readings_2.data", "sensor_readings_4.data", "sensor_readings_24.data")

def find_missing_files():
    return [f for f in REQUIRED_FILES if not list(DATA_DIRECTORY.rglob(f))]

def download_dataset():
    missing = find_missing_files()
    if not missing:
        print("Датасет уже загружен.")
        return
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print("Датасет из UCI ML Repository загружается...")

    with TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "wall_following_robot_navigation.zip"
        with urlopen(DATASET_URL, timeout=60) as response:
            with archive_path.open("wb") as archive_file:
                copyfileobj(response, archive_file)
            with ZipFile(archive_path) as archive:
                archive.extractall(DATA_DIRECTORY)

    missing = find_missing_files()
    if missing: 
        raise RuntimeError("После распаковки не найдены файлы: " + ", ".join(missing))
    print(f"Дата сохранен в {DATA_DIRECTORY}")

if __name__ == "__main__":
    download_dataset()