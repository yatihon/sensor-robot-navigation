from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
FEATURE_NAMES = {
    24: [f"US{i}" for i in range(1, 25)],
    4: ["front", "left", "right", "back"],
    2:["front", "left"],
}

def find_data_file(sensor_count):
    filename = f"sensor_readings_{sensor_count}.data"
    matches = list(DATA_DIRECTORY.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Файл {filename} не найден. "
            "Запустите: python src/download_data.py"
    )
    return matches[0]

def load_data(sensor_count=24):
    if sensor_count not in FEATURE_NAMES:
        raise ValueError("sensor_count должен быть равен 2, 4 или 24")
    file_path = find_data_file(sensor_count)
    columns = FEATURE_NAMES[sensor_count] + ["target"]

    dataframe = pd.read_csv(file_path, header=None, names=columns,)
    dataframe["target"] = dataframe["target"].astype(str).str.strip()
    x = dataframe.drop(columns="target")
    y = dataframe["target"]
    return x, y

if __name__ == "__main__":
    for sensor_count in (24, 4, 2):
        x, y = load_data(sensor_count)
        print(f"\n=== Набор с {sensor_count} признаками ===")
        print("Размер x:", x.shape)
        print("Размер y:", y.shape)
        print("\nРаспределение классов:")
        print(y.value_counts())