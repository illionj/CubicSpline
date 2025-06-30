#!/usr/bin/env python3
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

RAW_FILE = "xy.csv"
INT_FILE = "xy_interpolation.csv"
OUT_PNG = "xy_vs_interp.png"

root = pathlib.Path(__file__).resolve().parent
df_raw = pd.read_csv(root / RAW_FILE)
df_interp = pd.read_csv(root / INT_FILE)

plt.figure(figsize=(6, 6))
plt.scatter(df_raw["x"], df_raw["y"], color="red", label="raw points", zorder=3, s=20)
plt.plot(df_interp["x"], df_interp["y"], label="interpolated curve", linewidth=1)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("raw vs interpolated")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(root / OUT_PNG, dpi=150)
print(f"saved {OUT_PNG}")


