import pandas as pd
import numpy as np

# 1) remove only the trailing +0i / +0.0i / +0j / +0.0j
MCs['SNRinst_dB'] = (
    MCs['SNRinst_dB']
      .astype(str).str.strip()
      .str.replace(r'\+0(?:\.0+)?[ij]$', '', regex=True)  # ONLY drop +0i/+0j at the end
)

# 2) convert to float
MCs['SNRinst_dB'] = MCs['SNRinst_dB'].astype(float)

# 3) optional: round to 6 decimals for consistency
MCs['SNRinst_dB'] = MCs['SNRinst_dB'].round(6)

# quick sanity checks
print(MCs['SNRinst_dB'].dtype)          # float64
print(MCs['SNRinst_dB'].head()) 