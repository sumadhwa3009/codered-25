# Audiovisual Zooming

**"What You See Is What You Hear"** - GEV Beamformer (Nair et al., 2019)

## Files

| File | Description |
|------|-------------|
| `demo.m` | Quick test (2 sec, simple tones) |
| `audiovisual_zoom_compact.m` | Full simulation (3 sec, realistic signals) |

## Quick Start

```matlab
demo                          % Fast test
audiovisual_zoom_compact      % Full simulation
```

## Requirements

- MATLAB with **Signal Processing Toolbox**
- Check: run `ver` and look for "Signal Processing Toolbox"

## What It Does

1. Simulates 6-mic circular array
2. Target at 0°, interference at 90°
3. GEV beamformer focuses on ±15° FOV
4. Shows spectrograms + beam pattern
5. Plays input (noisy) then output (cleaned)

## Algorithm

```
STFT → Covariance → MPDR Power → Matrix Integration → GEV → ISTFT
```

## Expected Output

- **Suppression**: 15-25 dB at 90° interference
- **Beam pattern**: Main lobe at 0°, null toward 90°
