# Audiovisual Zooming Simulation

## "What You See Is What You Hear" - Implementation of Nair et al., 2019

This MATLAB implementation demonstrates the **Generalized Eigenvalue (GEV) beamforming** method for spatial audio filtering, allowing isolation of sound sources within a defined Field of View (FOV) while suppressing interference from other directions.

---

## Quick Start

### Synthetic Data Mode (Default)
```matlab
% Simply run the main script
audiovisual_zooming_simulation
```

### Real Data Mode
1. Record 6-channel audio from your Sipeed microphone array
2. Save as `array_input.wav` in the same directory
3. Edit the script:
```matlab
USE_REAL_DATA = true;
```
4. Run the script

---

## Algorithm Overview

### 1. Steering Vector Computation (Step A)
For a Uniform Circular Array (UCA), the steering vector represents the phase relationship between microphones for a plane wave from direction θ:

$$\nu_\theta(f) = [e^{-jk\mathbf{d}_1 \cdot \mathbf{u}_\theta}, ..., e^{-jk\mathbf{d}_M \cdot \mathbf{u}_\theta}]^T$$

Where:
- $k = 2\pi f/c$ is the wave number
- $\mathbf{d}_m$ is the position of microphone $m$
- $\mathbf{u}_\theta$ is the unit vector toward direction θ

### 2. Covariance Estimation (Step B)
The spatial covariance matrix is estimated from STFT frames:

$$\hat{R}(f) = \frac{1}{T} \sum_{t=1}^{T} \mathbf{x}(f,t)\mathbf{x}^H(f,t)$$

### 3. MPDR Power Estimation (Step C)
The Minimum Power Distortionless Response estimates spatial power distribution:

$$P(\theta) = [\nu_\theta^H \hat{R}^{-1} \nu_\theta]^{-1}$$

### 4. Matrix Integration (Step D)
Signal and noise covariance matrices are constructed by integrating over respective angular regions:

- **Signal Matrix (FOV)**: $R_s = \sum_{\theta \in FOV} P(\theta) \nu_\theta \nu_\theta^H$
- **Noise Matrix (Outside FOV)**: $R_n = \sum_{\theta \notin FOV} P(\theta) \nu_\theta \nu_\theta^H$

### 5. Generalized Eigenvalue Problem (Step E)
Find optimal beamformer weights by solving:

$$R_s \mathbf{w} = \lambda R_n \mathbf{w}$$

The eigenvector with the largest eigenvalue maximizes the Signal-to-Noise Ratio.

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_REAL_DATA` | `false` | Toggle synthetic/real data |
| `fs` | 16000 Hz | Sampling frequency |
| `num_mics` | 6 | Number of microphones |
| `array_radius` | 0.05 m | UCA radius |
| `fft_size` | 512 | STFT window size |
| `freq_min` | 300 Hz | Lower processing bound |
| `freq_max` | 3400 Hz | Upper processing bound |
| `target_FOV` | [-15, +15]° | Field of view for target |
| `angular_resolution` | 20° | Scanning resolution |

---

## Output Files

| File | Description |
|------|-------------|
| `zoomed_output.wav` | Enhanced audio (interference suppressed) |
| `input_reference.wav` | Original noisy input (reference mic) |

---

## Visualization Outputs

The script generates two figures:

### Figure 1: Main Results
1. **Array Geometry** - 6-mic UCA layout with target/interference directions
2. **Input Spectrogram** - Original noisy signal
3. **Output Spectrogram** - Enhanced signal after beamforming
4. **Time Domain** - Waveform comparison
5. **Beam Pattern (Cartesian)** - Directional gain in dB
6. **Beam Pattern (Polar)** - Polar representation

### Figure 2: Spatial Analysis
1. **MPDR Spatial Power Spectrum** - Direction-frequency power map
2. **Power Spectral Density** - Frequency domain comparison

---

## Hardware Setup for Real Data

### Sipeed 6-Mic Array Connection
1. Connect Sipeed array to Raspberry Pi / ESP32
2. Configure for 6-channel recording at 16 kHz
3. Record scenario:
   - **Target**: Person speaking at 0° (front)
   - **Interference**: Music/noise at 90° (side)
4. Save as 6-channel WAV file

### Recording Tips
- Ensure 10+ seconds of recording
- Maintain consistent source positions
- Avoid clipping (keep levels below 0 dBFS)
- Record in moderately reverberant environment

---

## Frequency Range Selection

The processing is restricted to **300 Hz - 3400 Hz** for two reasons:

1. **Spatial Aliasing**: Above ~3400 Hz, the 5cm array experiences aliasing
   - Maximum unambiguous frequency: $f_{max} = c/(2d) ≈ 3430$ Hz
   
2. **Low-Frequency Noise**: Below 300 Hz, environmental noise dominates and spatial resolution is poor

---

## Mathematical Details

### Why Generalized Eigenvalue?

The GEV approach maximizes the ratio of signal power to noise power:

$$\text{SNR} = \frac{\mathbf{w}^H R_s \mathbf{w}}{\mathbf{w}^H R_n \mathbf{w}}$$

This is superior to simple MVDR/MPDR because:
1. It explicitly models the spatial distribution of interference
2. It uses the estimated power spectrum to weight contributions
3. It provides optimal suppression for the specific noise field

### Regularization

A small regularization term ($\epsilon = 10^{-6}$) is added to covariance matrices:
- Ensures numerical stability
- Prevents ill-conditioning from limited data
- Provides robustness against model mismatch

---

## Troubleshooting

### Common Issues

1. **"File not found" error**
   - Ensure `array_input.wav` is in MATLAB's current directory
   - Check file has exactly 6 channels

2. **Poor suppression**
   - Verify source directions match configuration
   - Increase angular resolution (decrease `angular_resolution`)
   - Check for reverberation issues

3. **Numerical warnings**
   - Increase `reg_param` if matrices are singular
   - Ensure sufficient data frames for covariance estimation

4. **Audio artifacts**
   - Adjust `fft_size` and `hop_size`
   - Check for synthesis window normalization issues

---

## References

1. Nair, V., et al. (2019). "Audiovisual Zooming: What You See Is What You Hear"
2. Van Trees, H.L. (2002). "Optimum Array Processing"
3. Benesty, J., et al. (2008). "Microphone Array Signal Processing"

---

## License

This implementation is for educational and research purposes.

