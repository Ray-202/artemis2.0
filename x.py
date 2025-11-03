import numpy as np
import pandas as pd

def clean_snr_db(series: pd.Series, decimals: int = 6, mode: str = "keep_real", tol: float = 1e-12):
    """
    Clean SNR in dB that may be stored as complex-number strings like '93.3+0i' or '60.33+13.64i'.

    mode:
      - "keep_real": keep real(z) regardless of imag(z); good for EDA
      - "drop_imag": set to NaN when |imag(z)| > tol (strict)
    tol: tolerance to consider the imaginary part effectively zero.
    """
    s = series.astype(str).str.strip()
    # normalize formatting
    s = s.str.replace(',', '', regex=False)          # remove thousands separators if any
    s = s.str.replace(r'\s+', '', regex=True)        # remove spaces

    # Fast path: plain real numbers -> float
    is_plain_real = s.str.match(r'^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$', na=False)
    out = pd.Series(index=s.index, dtype='float64')
    out.loc[is_plain_real] = pd.to_numeric(s.loc[is_plain_real], errors='coerce')

    # Complex or odd strings -> parse as complex and handle imag part
    need_complex = ~is_plain_real
    def _parse_complex(v: str):
        try:
            z = complex(v.replace('i', 'j'))  # Python uses 'j'
            if mode == "drop_imag" and abs(z.imag) > tol:
                return np.nan
            return float(z.real)               # keep only the real dB value
        except Exception:
            return np.nan

    out.loc[need_complex] = s.loc[need_complex].map(_parse_complex)

    if decimals is not None:
        out = out.round(decimals)
    return out

# ---- apply to your dataframe ----
MCs['SNRinst_dB'] = clean_snr_db(MCs['SNRinst_dB'], decimals=6, mode="keep_real")  # or mode="drop_imag"

# Audit & report
def count_nonzero_imag(series: pd.Series, tol: float = 1e-12):
    s = series.astype(str).str.strip().str.replace(',', '', regex=False)
    def _imag_mag(v):
        try:
            return abs(complex(v.replace('i','j')).imag)
        except Exception:
            return 0.0
    mask_complex = s.str.contains('[ij]', regex=True)
    imag_vals = s.loc[mask_complex].map(_imag_mag)
    return (imag_vals > tol).sum(), int(mask_complex.sum())

n_nonzero, n_complex = count_nonzero_imag(MCs['SNRinst_dB'])
print(f"Complex-form entries: {n_complex}, with non-zero imag: {n_nonzero}")
print(MCs['SNRinst_dB'].dtype, MCs['SNRinst_dB'].describe())