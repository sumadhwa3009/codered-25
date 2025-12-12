# Audiovisual Zooming - Compact

**"What You See Is What You Hear"** - GEV Beamformer Implementation

## Quick Start

```matlab
audiovisual_zoom_compact
```

## What It Does

- Generates synthetic target (0°) + interference (90°)
- Runs GEV beamformer with ±15° FOV
- Shows spectrograms, beam patterns
- Plays input (noisy) then output (zoomed)
- Saves `in_noisy.wav` and `out_zoomed.wav`

## Requirements

- MATLAB with **Signal Processing Toolbox**

## Algorithm (per frequency bin)

1. **STFT** → time-frequency representation
2. **Covariance** → spatial statistics
3. **MPDR** → power vs direction
4. **Matrix Integration** → R_s (FOV) / R_n (outside)
5. **GEV** → solve R_s·w = λ·R_n·w → max eigenvector
6. **ISTFT** → back to time domain

## Using Real Data

Edit `audiovisual_zoom_compact.m`:
- Replace synthetic signal generation with:
```matlab
[mic_sig, fs] = audioread('array_input.wav');
N = size(mic_sig, 1);
```
- Place 6-channel `array_input.wav` in same folder
