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