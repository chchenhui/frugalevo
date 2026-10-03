import numpy as np

def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError("input shorter than window_size")
    return np.convolve(x, np.ones(window_size)/window_size, mode="valid")

def _reflect_pad(x, r):
    return np.pad(x, (r, r), mode="reflect") if r > 0 else x

def _gauss(x, sigma):
    r = max(1, int(np.ceil(3*sigma)))
    z = np.arange(-r, r+1, dtype=float)
    k = np.exp(-0.5*(z/sigma)**2)
    k /= k.sum()
    return np.convolve(_reflect_pad(x, r), k, mode="valid")

def _reversal_count(y):
    d = np.diff(y)
    return int(np.sum(d[:-1]*d[1:] < 0)) if len(d) > 1 else 0

def enhanced_filter_with_trend_preservation(x, window_size=20):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")
    if window_size < 3:
        return x.copy()
    if not np.all(np.isfinite(x)):
        f = x[np.isfinite(x)]
        v = float(np.median(f)) if f.size else 0.0
        x = np.nan_to_num(x, nan=v, posinf=v, neginf=v)

    d = np.diff(x)
    c = float(np.median(d)) if d.size else 0.0
    sc = 1.4826*float(np.median(np.abs(d-c))) if d.size else 0.0
    sc = max(sc, 1e-7)

    repaired = x.copy()
    if n >= 3:
        pred = 0.5*(x[:-2] + x[2:])
        bad = np.abs(x[1:-1]-pred) > 4.0*sc
        mid = repaired[1:-1]
        mid[bad] = pred[bad]
        repaired[1:-1] = mid

    try:
        from scipy.interpolate import UnivariateSpline
        t = np.arange(n, dtype=float)
        s = max(n*sc*sc*0.55, 1e-10)
        z = np.asarray(UnivariateSpline(t, repaired, s=s, k=3, ext=3)(t))
        r = repaired-z
        rc = float(np.median(r))
        rs = max(1.4826*float(np.median(np.abs(r-rc))), 1e-7)

        # Smooth Huber-like influence limiting, less discontinuous than hard gating.
        cap = 2.2*rs
        q = np.minimum(1.0, cap/np.maximum(np.abs(r-rc), 1e-12))
        winsor = z + (r-rc)*q + rc
        rz = np.asarray(UnivariateSpline(t, winsor, s=s, k=3, ext=3)(t))

        mae0 = np.mean(np.abs(repaired-z))+1e-12
        mae1 = np.mean(np.abs(repaired-rz))
        selected = rz if (_reversal_count(rz) <= _reversal_count(z) and
                          mae1 <= 1.02*mae0 and np.all(np.isfinite(rz))) else z
    except Exception:
        selected = _gauss(repaired, max(0.8, min(1.5, window_size/18.0)))

    return np.asarray(selected[window_size-1:], dtype=float)

def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    return adaptive_filter(input_signal, window_size)

def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, length)
    clean = (2*np.sin(2*np.pi*.5*t) + 1.5*np.sin(2*np.pi*2*t) +
             .5*np.sin(2*np.pi*5*t) + .8*np.exp(-t/5)*np.sin(2*np.pi*1.5*t))
    clean += .1*t*np.sin(.2*t) + np.cumsum(rng.randn(length)*.05)
    return clean+rng.normal(0, noise_level, length), clean

def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean = generate_test_signal(signal_length, noise_level)
    else:
        noisy_signal = np.asarray(noisy_signal, dtype=float)
        clean = None
    y = process_signal(noisy_signal, window_size, "enhanced")
    out = {"filtered_signal": y, "clean_signal": None, "noisy_signal": None,
           "correlation": 0.0, "noise_reduction": 0.0,
           "signal_length": len(y)}
    if clean is not None:
        delay = window_size-1
        m = min(len(y), len(clean)-delay)
        yy = y[:m]
        cc = clean[delay:delay+m]
        nn = noisy_signal[delay:delay+m]
        out["filtered_signal"] = yy
        out["clean_signal"] = cc
        out["noisy_signal"] = nn
        out["correlation"] = float(np.corrcoef(yy, cc)[0,1]) if m > 1 else 0.0
        before = np.var(nn-cc)
        after = np.var(yy-cc)
        out["noise_reduction"] = float((before-after)/before) if before > 0 else 0.0
        out["signal_length"] = m
    return out

if __name__ == "__main__":
    r = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {r['correlation']:.3f}")
    print(f"Noise reduction: {r['noise_reduction']:.3f}")
    print(f"Processed signal length: {r['signal_length']}")