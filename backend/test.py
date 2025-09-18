import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# 1) Lightkurve: fetch & clean
# ---------------------------
def fetch_flattened_lc(target, mission="Kepler", author=None, min_pts=2000):
    """
    Download all available light curves for a target, stitch, remove NaNs,
    flatten (detrend), and return time (days) & normalized flux (unitless).
    """
    sr = lk.search_lightcurve(target, mission=mission, author=author)  # e.g., author="Kepler" or "SPOC"
    if len(sr) == 0:
        raise ValueError(f"No light curves found for {target} in {mission}.")
    lc = sr.download_all().stitch().remove_nans()
    # Use PDC when available; flatten to remove variability/systematics.
    lc = lc.flatten(window_length=401, polyorder=2)
    # Normalize around median, then standardize for ML stability
    f = lc.flux.value.astype(np.float32)
    t = lc.time.value.astype(np.float32)
    if len(f) < min_pts:
        raise ValueError(f"Too few cadences for {target} ({len(f)}).")
    f = f / np.nanmedian(f)
    f = (f - np.nanmedian(f)) / (np.nanstd(f) + 1e-6)
    return t, f

# ---------------------------
# 2) BLS + phase folding
# ---------------------------
def bls_best(t, f, pmin=0.5, pmax=20.0, qmin=0.0005, qmax=0.1):
    """Run Box Least Squares and return (period, duration, t0)."""
    bls = BoxLeastSquares(t, f)
    periods = np.linspace(pmin, pmax, 5000).astype(np.float32)
    durations = np.linspace(qmin, qmax, 20).astype(np.float32)
    power = bls.power(periods, durations)
    i = int(np.nanargmax(power.power))
    return float(power.period[i]), float(power.duration[i]), float(power.transit_time[i])

def fold_to_bins(t, f, period, t0, nbins=512):
    """Phase fold and median-bin into fixed-length vector."""
    phase = ((t - t0 + 0.5 * period) % period) / period  # center transit ~0.5
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(phase, bins) - 1
    pf = np.array([np.nanmedian(f[idx == k]) for k in range(nbins)], dtype=np.float32)
    # standardize
    pf = (pf - np.nanmedian(pf)) / (np.nanstd(pf) + 1e-6)
    return pf

# ---------------------------
# 3) Simple transit injection (box model)
# ---------------------------
def inject_box_transit(t, base_flux, period, t0, duration_days, depth_frac):
    """Multiply base flux by a simple box-shaped transit (depth_frac ~ 1e-4 .. 1e-2)."""
    phase = ((t - t0) % period) / period
    in_transit = (phase < duration_days / period).astype(np.float32)
    model = 1.0 - depth_frac * in_transit
    return (base_flux * model).astype(np.float32)

# ---------------------------
# 4) Dataset built from real LCs via Lightkurve
#    - negatives: original LC, folded on its BLS (often flat)
#    - positives: same LC with injected transit, folded on injected period
# ---------------------------
class FoldedDataset(Dataset):
    def __init__(self, targets, mission="Kepler", nbins=512,
                 neg_per_target=1, pos_per_target=2,
                 p_range=(0.6, 12.0), dur_frac_range=(0.015, 0.10), depth_range_ppm=(200, 5000)):
        self.X, self.y = [], []
        for target in targets:
            try:
                t, f = fetch_flattened_lc(target, mission=mission)
            except Exception as e:
                print("skip", target, e); continue

            # ---- negatives (no injection); fold on its own BLS pick
            for _ in range(neg_per_target):
                try:
                    Pn, Dn, t0n = bls_best(t, f)
                    self.X.append(fold_to_bins(t, f, Pn, t0n, nbins))
                    self.y.append(0)
                except Exception as e:
                    print("neg fail", target, e)

            # ---- positives (inject → fold on injected period)
            for _ in range(pos_per_target):
                P = np.random.uniform(*p_range)
                dur = np.random.uniform(*dur_frac_range) * P
                depth = np.random.uniform(depth_range_ppm[0], depth_range_ppm[1]) * 1e-6
                t0 = np.random.uniform(0, P)
                finj = inject_box_transit(t, f, P, t0, dur, depth)
                self.X.append(fold_to_bins(t, finj, P, t0, nbins))
                self.y.append(1)

        self.X = np.stack(self.X).astype(np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        print(f"Built dataset: X={self.X.shape}, positives={int(self.y.sum())}, negatives={int((1-self.y).sum())}")

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

# ---------------------------
# 5) Tiny 1D-CNN
# ---------------------------
class SmallCNN(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):              # x: (B, L)
        x = x.unsqueeze(1)             # (B, 1, L)
        f = self.feature(x)            # (B, 128, 1)
        return self.head(f).squeeze(1) # (B,)

# ---------------------------
# 6) Train
# ---------------------------
def train_detector(targets, mission="Kepler", nbins=512, epochs=8, batch=256, pos_mult=2):
    ds = FoldedDataset(targets, mission=mission, nbins=nbins,
                       neg_per_target=1, pos_per_target=pos_mult)
    n = len(ds); n_val = max(256, int(0.2 * n))
    n_tr = n - n_val
    ds_tr, ds_val = torch.utils.data.random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(nbins).to(device)
    pos_frac = float(ds.y.mean())
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * X.size(0)

        # Validation AUPRC
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for X, y in dl_val:
                X = X.to(device)
                p = torch.sigmoid(model(X)).cpu().numpy()
                ps.append(p); ys.append(y.numpy())
        ps = np.concatenate(ps); ys = np.concatenate(ys)
        auprc = average_precision_score(ys, ps)
        print(f"Epoch {ep:02d} | train loss {tot/n_tr:.4f} | AUPRC {auprc:.3f}")

    return model, ds

# ---------------------------
# 7) Demo: train & score a real target
# ---------------------------
if __name__ == "__main__":
    # Start with a few well-observed Kepler and TESS targets (add more for better training)
    training_targets = [
        "Kepler-10", "Kepler-8", "Kepler-12",  # Kepler examples
        "Pi Mensae", "WASP-18", "HD 209458"    # bright TESS targets
    ]
    model, ds = train_detector(training_targets, mission="Kepler", nbins=512, epochs=8, batch=256, pos_mult=2)

    # Score a brand-new star (Lightkurve fetch → BLS → fold → model score)
    target_to_score = "Kepler-10"
    t, f = fetch_flattened_lc(target_to_score, mission="Kepler")
    # Use BLS to find the *star's* best period (may or may not be a planet)
    P, D, t0 = bls_best(t, f, pmin=0.5, pmax=20.0)
    pf = fold_to_bins(t, f, P, t0, nbins=512)
    with torch.no_grad():
        x = torch.from_numpy(pf).float().unsqueeze(0)
        score = torch.sigmoid(model(x)).item()
    print(f"{target_to_score} → planet-like score: {score:.3f}")
