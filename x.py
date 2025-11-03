import numpy as np
import pandas as pd

def clean_snr_db_strict(series: pd.Series, decimals: int = 6, mode: str = "keep_real", tol: float = 1e-12):
    """
    Clean SNR dB column with possible:
      - scientific notation (e.g., 6.03e+01)
      - corrupted sci-notation where 'e' vanished (e.g., 6.03+01)  -> we restore 'e'
      - complex strings with i/j (e.g., 60.33+13.64i)

    mode:
      - 'keep_real': return real(z) for complex strings
      - 'drop_imag': set NaN when |imag(z)| > tol
    """
    s = series.astype(str).str.strip()

    # 1) normalize commas and spaces
    s = s.str.replace(',', '', regex=False).str.replace(r'\s+', '', regex=True)

    # 2) First, handle the “missing e” case ONLY when there is no i/j
    #    Pattern: ...digits[.digits][+/-]digits AT THE VERY END and no 'e' or 'E' present
    no_e_mask = ~s.str.contains('[eEij]', regex=True) & s.str.contains(r'[+-]\d+$', regex=True)
    s.loc[no_e_mask] = s.loc[no_e_mask].str.replace(r'([0-9.])([+-]\d+)$', r'\1e\2', regex=True)

    # 3) Fast path: plain real (including proper scientific notation)
    is_plain_real = s.str.match(r'^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$', na=False)

    out = pd.Series(index=s.index, dtype='float64')
    out.loc[is_plain_real] = pd.to_numeric(s.loc[is_plain_real], errors='coerce')

    # 4) Complex or odd forms -> parse as complex and handle imag part
    need_complex = ~is_plain_real
    def _parse_complex(v: str):
        try:
            z = complex(v.replace('i','j'))
            if mode == "drop_imag" and abs(z.imag) > tol:
                return np.nan
            return float(z.real)
        except Exception:
            return np.nan

    out.loc[need_complex] = s.loc[need_complex].map(_parse_complex)

    if decimals is not None:
        out = out.round(decimals)
    return out

# ---- Apply ONLY to the SNR column (from a fresh/original copy if possible) ----
# e.g., reload just this column from the CSV if you can:
# MCs['SNRinst_dB_raw'] = pd.read_csv(PATH, usecols=['SNRinst_dB'])['SNRinst_dB']

MCs['SNRinst_dB'] = clean_snr_db_strict(MCs['SNRinst_dB'], decimals=6, mode="keep_real")

print(MCs['SNRinst_dB'].dtype)
print(MCs['SNRinst_dB'].head())




# 0) original kept intact
mcs_raw = mcs.copy(deep=True)      # BEFORE

# 1) working copy to clean
mcs_clean = mcs_raw.copy(deep=True)  # AFTER
mcs_clean['SNRinst_dB'] = clean_snr_db_strict(
    mcs_clean['SNRinst_dB'], decimals=6, mode="keep_real"
)

# 2) Compare
print("Raw dtype:",   mcs_raw['SNRinst_dB'].dtype)
print("Clean dtype:", mcs_clean['SNRinst_dB'].dtype)

# Show a few rows side-by-side (sample to keep output small)
idx = mcs_clean.sample(10, random_state=42).index
display(
    mcs_raw.loc[idx, ['targetID','Object','Size','SNRinst_dB']].rename(columns={'SNRinst_dB':'SNR_raw'})
    .join(mcs_clean.loc[idx, ['SNRinst_dB']].rename(columns={'SNRinst_dB':'SNR_clean'}))
)

# 3) Summary stats (avoid huge prints)
print("Raw describe (SNR):\n", mcs_raw['SNRinst_dB'].astype(str).head())  # strings
print("Clean describe (SNR):\n", mcs_clean['SNRinst_dB'].describe())




# 1) Count non-zero imaginary cases in the raw column (audit)
def count_nonzero_imag(series, tol=1e-12):
    s = series.astype(str).str.strip()
    def imag_abs(x):
        try: return abs(complex(x.replace('i','j')).imag)
        except: return 0.0
    mask_cplx = s.str.contains('[ij]', regex=True)
    n_complex = int(mask_cplx.sum())
    n_nonzero = int((s.loc[mask_cplx].map(imag_abs) > tol).sum())
    return n_complex, n_nonzero

n_complex, n_nonzero = count_nonzero_imag(mcs_raw['SNRinst_dB'])
print(f"Raw SNR: complex-form entries={n_complex}, non-zero imag={n_nonzero}")

# 2) How many values actually changed (raw string vs cleaned float repr)
changed = (
    mcs_raw['SNRinst_dB'].astype(str).str.strip() !=
    mcs_clean['SNRinst_dB'].round(6).astype(str).str.strip()
).sum()
print("Rows different after cleaning:", int(changed))
