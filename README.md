# Audiovisual Zooming

**"What You See Is What You Hear"** - GEV Beamformer (Nair et al., 2019)

## Microphone Array

**7-mic configuration** (6 circular + 1 center):

| Mic | File | Position |
|-----|------|----------|
| mic0 | mic0_0deg.wav | Front (0°) |
| mic1 | mic1_60deg.wav | Front-right (60°) |
| mic2 | mic2_120deg.wav | Back-right (120°) |
| mic3 | mic3_180deg.wav | Back (180°) |
| mic4 | mic4_240deg.wav | Back-left (240°) |
| mic5 | mic5_300deg.wav | Front-left (300°) |
| mic6 | mic6_0deg.wav | Center |

## Files

| File | Description |
|------|-------------|
| `run_demo.m` | Quick test (synthetic signals) |
| `audiovisual_zoom_compact.m` | Full simulation |

## Quick Start

```matlab
run_demo                      % Fast demo
audiovisual_zoom_compact      % Full simulation
```

## Requirements

- MATLAB with **Signal Processing Toolbox**

## Algorithm

```
STFT → Covariance → MPDR Power → Matrix Integration → GEV → ISTFT
```

Target at 0°, Interference at 90° → **15-25 dB suppression**
