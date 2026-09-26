from pathlib import Path
import matplotlib.pyplot as plt

noise_lvls = [0, 1, 5, 10, 20]
f1_2features = [1.0000, 0.9913, 0.9383, 0.8635, 0.7633]
std_2features = [0.0000, 0.0018, 0.0070, 0.0096, 0.0120]
f1_24features = [0.9830, 0.9708, 0.8831, 0.7946, 0.6865]
std_24features = [0.0000, 0.0042, 0.0083, 0.0105, 0.0116]

plt.figure(figsize=(9, 6))
plt.errorbar(noise_lvls, f1_2features, yerr=std_2features, marker="o", capsize=4, linewidth=2, label="2 aggregated figures")
plt.errorbar(noise_lvls, f1_24features, yerr=std_24features, marker="o", capsize=4, linewidth=2, label="24 raw features")

plt.xlabel("Noise level, % of feature standard deviation")
plt.ylabel("Macro-F1")
plt.title("Model robustness to sensor noise")
plt.xticks(noise_lvls)
plt.ylim(0.6, 1.02)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

project_root = Path(__file__).resolve().parents[1]
figure_directory = project_root / "reports" / "figures"
figure_directory.mkdir(parents=True, exist_ok=True)
figure_path = figure_directory / "noise_robustness.png"

plt.savefig(figure_path, dpi=150)
plt.close()
print(f"График сохранен: {figure_path}")
