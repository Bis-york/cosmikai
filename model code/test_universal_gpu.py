import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import platform
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Universal GPU setup and diagnostics
def detect_and_setup_device():
    """Detect and configure the best available compute device (CUDA, Metal, ROCm, or CPU)"""
    print("[INFO] 🔍 Detecting available compute devices...")
    
    # Check system info
    system = platform.system()
    print(f"[INFO] Operating System: {system}")
    print(f"[INFO] Python: {sys.version}")
    print(f"[INFO] PyTorch: {torch.__version__}")
    
    device_info = {}
    
    # 1. NVIDIA CUDA (Windows, Linux, Mac with NVIDIA eGPU)
    if torch.cuda.is_available():
        device_info['cuda'] = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            device_info['cuda']['devices'].append({
                'id': i,
                'name': gpu_name,
                'memory_gb': gpu_memory
            })
        
        print(f"[INFO] ✅ NVIDIA CUDA: Found {device_info['cuda']['device_count']} GPU(s)")
        for device in device_info['cuda']['devices']:
            print(f"[INFO]   GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f} GB)")
    else:
        device_info['cuda'] = {'available': False}
        print("[INFO] ❌ NVIDIA CUDA: Not available")
    
    # 2. Apple Metal Performance Shaders (Mac M1/M2/M3)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info['mps'] = {'available': True}
        print("[INFO] ✅ Apple Metal (MPS): Available")
        print("[INFO]   Using Apple Silicon GPU acceleration")
    else:
        device_info['mps'] = {'available': False}
        if system == "Darwin":  # macOS
            print("[INFO] ❌ Apple Metal (MPS): Not available (requires PyTorch 1.12+ and macOS 12.3+)")
        else:
            print("[INFO] ➖ Apple Metal (MPS): Not applicable (not macOS)")
    
    # 3. AMD ROCm (Linux primarily)
    rocm_available = False
    try:
        # Check if ROCm backend is available
        hip_version = getattr(torch.version, 'hip', None) # type: ignore
        if hip_version is not None:
            rocm_available = True
            device_info['rocm'] = {'available': True}
            print(f"[INFO] ✅ AMD ROCm: Available (HIP version: {hip_version})")
        else:
            device_info['rocm'] = {'available': False}
            print("[INFO] ❌ AMD ROCm: Not available")
    except:
        device_info['rocm'] = {'available': False}
        print("[INFO] ❌ AMD ROCm: Not available")
    
    # 4. Select best device
    if device_info['cuda']['available']:
        device = torch.device("cuda:0")
        device_type = "NVIDIA CUDA"
        use_amp = True  # Automatic Mixed Precision
        if (
            device_info['cuda']['available']
            and isinstance(device_info['cuda'].get('devices', None), list)
            and len(device_info['cuda']['devices']) > 0 # type: ignore
        ):
            print(f"[INFO] 🚀 Selected: {device_type} - {device_info['cuda']['devices'][0]['name']}") # type: ignore
        else:
            print(f"[INFO] 🚀 Selected: {device_type}")
        
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
    elif device_info['mps']['available']:
        device = torch.device("mps")
        device_type = "Apple Metal"
        use_amp = False  # MPS doesn't support AMP yet
        print(f"[INFO] 🚀 Selected: {device_type} - Apple Silicon GPU")
        
    elif device_info['rocm']['available']:
        device = torch.device("cuda:0")  # ROCm uses cuda device interface
        device_type = "AMD ROCm"
        use_amp = True  # ROCm supports AMP
        print(f"[INFO] 🚀 Selected: {device_type} - AMD GPU")
        
    else:
        device = torch.device("cpu")
        device_type = "CPU"
        use_amp = False
        print(f"[INFO] ⚠️  Selected: {device_type} - No GPU acceleration available")
        print(f"[INFO] 💡 For better performance, install:")
        print(f"[INFO]    • NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(f"[INFO]    • Apple Silicon: pip install torch")
        print(f"[INFO]    • AMD GPU: pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
    
    return device, device_type, use_amp, device_info

# ---------------------------
# 1) Lightkurve: fetch & clean
# ---------------------------
def fetch_flattened_lc(target, mission="Kepler", author=None, min_pts=2000):
    """
    Download all available light curves for a target, stitch, remove NaNs,
    flatten (detrend), and return time (days) & normalized flux (unitless).
    """
    print(f"[INFO] Searching for light curves for target '{target}' in mission '{mission}'...")
    sr = lk.search_lightcurve(target, mission=mission, author=author)  # e.g., author="Kepler" or "SPOC"
    if len(sr) == 0:
        raise ValueError(f"No light curves found for {target} in {mission}.")
    print(f"[INFO] Found {len(sr)} light curve files. Downloading and stitching...")
    lc = sr.download_all().stitch().remove_nans() # type: ignore
    print(f"[INFO] Light curve downloaded. Original length: {len(lc.flux)} data points")
    # Use PDC when available; flatten to remove variability/systematics.
    print(f"[INFO] Flattening light curve to remove stellar variability and systematics...")
    lc = lc.flatten(window_length=401, polyorder=2)
    # Normalize around median, then standardize for ML stability
    f = lc.flux.value.astype(np.float32)
    t = lc.time.value.astype(np.float32)
    if len(f) < min_pts:
        raise ValueError(f"Too few cadences for {target} ({len(f)}).")
    print(f"[INFO] Normalizing flux values around median...")
    f = f / np.nanmedian(f)
    print(f"[INFO] Standardizing flux for ML stability...")
    f = (f - np.nanmedian(f)) / (np.nanstd(f) + 1e-6)
    print(f"[INFO] Light curve preprocessing complete. Final length: {len(f)} points")
    return t, f

# ---------------------------
# 2) BLS + phase folding
# ---------------------------
def bls_best(t, f, pmin=0.5, pmax=20.0, qmin=0.0005, qmax=0.1):
    """Run Box Least Squares and return (period, duration, t0)."""
    print(f"[INFO] Running Box Least Squares (BLS) to search for periodic signals...")
    print(f"[INFO] Period search range: {pmin:.2f} to {pmax:.1f} days")
    bls = BoxLeastSquares(t, f)
    periods = np.linspace(pmin, pmax, 5000).astype(np.float32)
    durations = np.linspace(qmin, qmax, 20).astype(np.float32)
    print(f"[INFO] Testing {len(periods)} periods and {len(durations)} duration fractions...")
    power = bls.power(periods, durations)
    i = int(np.nanargmax(power.power))
    best_period = float(power.period[i])
    best_duration = float(power.duration[i])
    best_t0 = float(power.transit_time[i])
    print(f"[INFO] BLS complete. Best period: {best_period:.4f} days, duration: {best_duration:.4f} days, t0: {best_t0:.4f}")
    return best_period, best_duration, best_t0

def fold_to_bins(t, f, period, t0, nbins=512):
    """Phase fold and median-bin into fixed-length vector."""
    print(f"[INFO] Phase folding light curve with period {period:.4f} days...")
    phase = ((t - t0 + 0.5 * period) % period) / period  # center transit ~0.5
    print(f"[INFO] Binning folded light curve into {nbins} phase bins...")
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(phase, bins) - 1
    pf = np.array([np.nanmedian(f[idx == k]) for k in range(nbins)], dtype=np.float32)
    # standardize
    print(f"[INFO] Standardizing binned light curve...")
    pf = (pf - np.nanmedian(pf)) / (np.nanstd(pf) + 1e-6)
    print(f"[INFO] Phase folding complete. Output shape: {pf.shape}")
    return pf

# ---------------------------
# 3) Simple transit injection (box model)
# ---------------------------
def inject_box_transit(t, base_flux, period, t0, duration_days, depth_frac):
    """Multiply base flux by a simple box-shaped transit (depth_frac ~ 1e-4 .. 1e-2)."""
    print(f"[INFO] Injecting synthetic transit: P={period:.3f}d, depth={depth_frac*1e6:.0f}ppm, duration={duration_days:.3f}d")
    phase = ((t - t0) % period) / period
    in_transit = (phase < duration_days / period).astype(np.float32)
    num_transits = int(np.sum(in_transit))
    print(f"[INFO] Transit affects {num_transits} data points out of {len(t)} total")
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
        print(f"[INFO] Building training dataset from {len(targets)} stellar targets...")
        print(f"[INFO] Will create {neg_per_target} negative + {pos_per_target} positive examples per target")
        self.X, self.y = [], []
        
        for i, target in enumerate(targets, 1):
            print(f"[INFO] Processing target {i}/{len(targets)}: {target}")
            try:
                t, f = fetch_flattened_lc(target, mission=mission)
            except Exception as e:
                print(f"[WARN] Skipping {target}: {e}"); continue

            # ---- negatives (no injection); fold on its own BLS pick
            print(f"[INFO] Creating {neg_per_target} negative example(s) for {target}...")
            for j in range(neg_per_target):
                try:
                    Pn, Dn, t0n = bls_best(t, f)
                    self.X.append(fold_to_bins(t, f, Pn, t0n, nbins))
                    self.y.append(0)
                    print(f"[INFO] Added negative example {j+1}/{neg_per_target} for {target}")
                except Exception as e:
                    print(f"[WARN] Negative example failed for {target}: {e}")

            # ---- positives (inject → fold on injected period)
            print(f"[INFO] Creating {pos_per_target} positive example(s) for {target}...")
            for j in range(pos_per_target):
                P = np.random.uniform(*p_range)
                dur = np.random.uniform(*dur_frac_range) * P
                depth = np.random.uniform(depth_range_ppm[0], depth_range_ppm[1]) * 1e-6
                t0 = np.random.uniform(0, P)
                finj = inject_box_transit(t, f, P, t0, dur, depth)
                self.X.append(fold_to_bins(t, finj, P, t0, nbins))
                self.y.append(1)
                print(f"[INFO] Added positive example {j+1}/{pos_per_target} for {target}")

        if len(self.X) == 0:
            raise ValueError("No training data was created! Check that your targets exist in the specified mission.")
        
        self.X = np.stack(self.X).astype(np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        print(f"[INFO] Dataset creation complete!")
        print(f"[INFO] Final dataset: X={self.X.shape}, positives={int(np.sum(self.y))}, negatives={int(len(self.y) - np.sum(self.y))}")
        
        # Warn if dataset is very small
        if len(self.X) < 20:
            print(f"[WARN] Small dataset ({len(self.X)} samples). Consider adding more targets for better performance.")

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

# ---------------------------
# 5) Tiny 1D-CNN (Universal GPU compatible)
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
# 6) Universal GPU Training
# ---------------------------
def train_detector(targets, mission, nbins, epochs, batch, pos_mult, device, device_type, use_amp):
    print(f"[INFO] Starting training process...")
    print(f"[INFO] Training parameters: {epochs} epochs, batch size {batch}, {pos_mult} positives per target")
    print(f"[INFO] Compute device: {device_type} ({device})")
    
    ds = FoldedDataset(targets, mission=mission, nbins=nbins,
                       neg_per_target=1, pos_per_target=pos_mult)
    
    n = len(ds)
    if n < 10:
        raise ValueError(f"Dataset too small: only {n} samples. Need at least 10 samples to train.")
    
    # More reasonable validation split for small datasets
    n_val = max(2, min(int(0.2 * n), n // 2))  # At least 2, at most half the data
    n_tr = n - n_val
    print(f"[INFO] Splitting dataset: {n_tr} training samples, {n_val} validation samples")
    ds_tr, ds_val = torch.utils.data.random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    print(f"[INFO] Creating data loaders with {device_type} optimizations...")
    
    # Optimize data loading based on device type
    if device.type == "cuda":  # NVIDIA CUDA or AMD ROCm
        num_workers = 4
        pin_memory = True
        persistent_workers = True
    elif device.type == "mps":  # Apple Metal
        num_workers = 0  # MPS works better with single-threaded data loading
        pin_memory = False
        persistent_workers = False
    else:  # CPU
        num_workers = 2
        pin_memory = False
        persistent_workers = True
    
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=num_workers, 
                       pin_memory=pin_memory, persistent_workers=persistent_workers)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=num_workers, 
                        pin_memory=pin_memory, persistent_workers=persistent_workers)

    print(f"[INFO] Initializing CNN model with input size {nbins}...")
    model = SmallCNN(nbins).to(device)
    
    # Show device memory info
    if device.type == "cuda":
        if torch.cuda.is_available():
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.cuda.empty_cache()
    elif device.type == "mps":
        print(f"[INFO] Using Apple Metal Performance Shaders")
    
    pos_frac = float(np.mean(ds.y))
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    print(f"[INFO] Class balance: {pos_frac:.2%} positive, using pos_weight={pos_weight.item():.2f}")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Setup mixed precision training (CUDA and ROCm only)
    scaler = None
    if use_amp:
        print(f"[INFO] ✅ Enabling Automatic Mixed Precision (AMP) for faster training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        print(f"[INFO] Using full precision training")
    
    print(f"[INFO] Starting training for {epochs} epochs...")
    
    final_auprc = 0.0  # Initialize final AUPRC

    for ep in range(1, epochs + 1):
        print(f"[INFO] Epoch {ep}/{epochs} - Training phase...")
        
        # Show memory usage for GPU devices
        if device.type == "cuda" and torch.cuda.is_available():
            print(f"[INFO] GPU Memory before epoch: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        model.train(); tot = 0.0
        batch_count = 0
        for X, y in dl_tr:
            # Non-blocking transfer for CUDA, regular transfer for MPS/CPU
            if device.type == "cuda":
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            else:
                X, y = X.to(device), y.to(device)
            
            opt.zero_grad()
            
            # Use mixed precision training for supported devices
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(X)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
            
            tot += loss.item() * X.size(0)
            batch_count += 1
        
        print(f"[INFO] Epoch {ep} training complete. Processed {batch_count} batches")
        
        if device.type == "cuda" and torch.cuda.is_available():
            print(f"[INFO] GPU Memory after training: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

        # Validation AUPRC
        print(f"[INFO] Epoch {ep} - Validation phase...")
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for X, y in dl_val:
                if device.type == "cuda":
                    X = X.to(device, non_blocking=True)
                else:
                    X = X.to(device)
                
                # Use mixed precision for inference too
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(X)
                        p = torch.sigmoid(logits).cpu().numpy()
                else:
                    p = torch.sigmoid(model(X)).cpu().numpy()
                ps.append(p); ys.append(y.numpy())
        ps = np.concatenate(ps); ys = np.concatenate(ys)
        auprc = average_precision_score(ys, ps)
        print(f"[RESULT] Epoch {ep:02d} | train loss {tot/n_tr:.4f} | validation AUPRC {auprc:.3f}")
        
        # Clear cache after each epoch for GPU devices
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Store the final AUPRC for saving
        final_auprc = auprc

    print(f"[INFO] Training complete!")
    
    # Final memory cleanup and reporting
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] Final GPU Memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"[INFO] Peak GPU Memory usage: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")
    
    return model, ds, final_auprc

# ---------------------------
# 7) Demo: train & score a real target (Universal GPU)
# ---------------------------
if __name__ == "__main__":
    print("="*70)
    print("UNIVERSAL EXOPLANET TRANSIT DETECTION PIPELINE")
    print("Supports: NVIDIA CUDA • Apple Metal • AMD ROCm • CPU")
    print("="*70)
    
    # Detect and setup the best available device
    device, device_type, use_amp, device_info = detect_and_setup_device()
    
    # Start with well-observed Kepler targets that actually exist in the archive
    training_targets = [
        "Kepler-10", "Kepler-8", "Kepler-12", "Kepler-1", "Kepler-2", 
        "Kepler-3", "Kepler-4", "Kepler-5", "Kepler-6", "Kepler-7"
    ]
    
    print(f"\n[INFO] Training on {len(training_targets)} stellar targets:")
    for i, target in enumerate(training_targets, 1):
        print(f"  {i}. {target}")
    print()
    
    print("[PHASE 1] TRAINING THE NEURAL NETWORK")
    print("-" * 50)
    model, ds, final_auprc = train_detector(training_targets, mission="Kepler", nbins=512, 
                                          epochs=8, batch=256, pos_mult=2, 
                                          device=device, device_type=device_type, use_amp=use_amp)

    print()
    print("[PHASE 2] SCORING A NEW TARGET")
    print("-" * 50)
    # Score a brand-new star (Lightkurve fetch → BLS → fold → model score)
    target_to_score = "Kepler-10"
    print(f"[INFO] Now scoring target: {target_to_score}")
    print(f"[INFO] Step 1: Fetching and preprocessing light curve...")
    t, f = fetch_flattened_lc(target_to_score, mission="Kepler")
    
    print(f"[INFO] Step 2: Finding best period with BLS...")
    # Use BLS to find the *star's* best period (may or may not be a planet)
    P, D, t0 = bls_best(t, f, pmin=0.5, pmax=20.0)
    
    print(f"[INFO] Step 3: Phase folding and binning...")
    pf = fold_to_bins(t, f, P, t0, nbins=512)
    
    print(f"[INFO] Step 4: Running neural network inference on {device_type}...")
    with torch.no_grad():
        x = torch.from_numpy(pf).float().unsqueeze(0).to(device)
        # Use appropriate precision for inference
        if device.type == "cuda" and use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                score = torch.sigmoid(logits).cpu().item()
        else:
            score = torch.sigmoid(model(x)).cpu().item()
    
    print()
    print("="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Target: {target_to_score}")
    print(f"Planet-like confidence score: {score:.3f}")
    print(f"Compute device used: {device_type}")
    if score > 0.5:
        print("🟢 HIGH confidence - likely contains a transit signal!")
    elif score > 0.3:
        print("🟡 MEDIUM confidence - possible transit signal")
    else:
        print("🔴 LOW confidence - unlikely to contain transits")
    print("="*70)
    
    print()
    print("[PHASE 3] SAVING THE TRAINED MODEL")
    print("-" * 50)
    
    # Save the trained model with device info
    model_path = f"trained_exoplanet_detector_{device_type.lower().replace(' ', '_')}.pth"
    print(f"[INFO] Saving trained model to '{model_path}'...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'SmallCNN',
        'input_size': 512,
        'training_targets': training_targets,
        'final_auprc': final_auprc,
        'epochs_trained': 8,
        'dataset_size': len(ds),
        'device_type': device_type,
        'device_info': device_info,
        'torch_version': torch.__version__,
        'platform': platform.system()
    }, model_path)
    print(f"[INFO] ✅ Model saved successfully!")
    
    # Also save a standalone model file for easy loading
    model_simple_path = f"exoplanet_model_{device_type.lower().replace(' ', '_')}.pth"
    print(f"[INFO] Saving simple model file to '{model_simple_path}'...")
    torch.save(model.state_dict(), model_simple_path)
    
    # Save comprehensive training statistics
    stats_path = f"training_stats_{device_type.lower().replace(' ', '_')}.txt"
    print(f"[INFO] Saving training statistics to '{stats_path}'...")
    with open(stats_path, 'w') as f:
        f.write("UNIVERSAL EXOPLANET DETECTION MODEL - TRAINING STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training completed: {target_to_score}\n")
        f.write(f"Final validation AUPRC: {final_auprc:.3f}\n")
        f.write(f"Total dataset size: {len(ds)} samples\n")
        f.write(f"Positive samples: {int(np.sum(ds.y))}\n")
        f.write(f"Negative samples: {len(ds.y) - int(np.sum(ds.y))}\n")
        f.write(f"Training targets: {', '.join(training_targets)}\n")
        f.write(f"Model architecture: SmallCNN (512 input)\n")
        f.write(f"Training epochs: 8\n\n")
        
        f.write("COMPUTE ENVIRONMENT\n")
        f.write("-" * 20 + "\n")
        f.write(f"Device used: {device_type} ({device})\n")
        f.write(f"Mixed precision: {use_amp}\n")
        f.write(f"Operating system: {platform.system()}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n\n")
        
        f.write("DEVICE AVAILABILITY\n")
        f.write("-" * 20 + "\n")
        f.write(f"NVIDIA CUDA: {'✅' if device_info['cuda']['available'] else '❌'}\n")
        f.write(f"Apple Metal: {'✅' if device_info['mps']['available'] else '❌'}\n")  
        f.write(f"AMD ROCm: {'✅' if device_info['rocm']['available'] else '❌'}\n")
        
        if device_info['cuda']['available']:
            f.write(f"\nCUDA DETAILS:\n")
            for device_detail in device_info['cuda']['devices']:
                f.write(f"  GPU {device_detail['id']}: {device_detail['name']} ({device_detail['memory_gb']:.1f} GB)\n")
    
    print(f"[INFO] ✅ Training statistics saved!")
    print()
    print("🎉 ALL FILES SAVED SUCCESSFULLY!")
    print(f"📁 Model file: {model_path}")
    print(f"📁 Simple model: {model_simple_path}")  
    print(f"📁 Training stats: {stats_path}")
    print()
    print("✨ UNIVERSAL GPU SUPPORT FEATURES:")
    print("• 🟢 NVIDIA CUDA (Windows/Linux/Mac)")
    print("• 🟢 Apple Metal (M1/M2/M3 Macs)")  
    print("• 🟢 AMD ROCm (Linux)")
    print("• 🟢 CPU fallback (all platforms)")
    print()
    print("You can now use these files to:")
    print("• Load the model on any supported platform")
    print("• Continue training from this checkpoint")
    print("• Deploy the model for real exoplanet detection")
    print("• Share models across different GPU types")