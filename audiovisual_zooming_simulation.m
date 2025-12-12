%% ========================================================================
%  AUDIOVISUAL ZOOMING SIMULATION
%  "What You See Is What You Hear" - Nair et al., 2019
%  ========================================================================
%  This script implements the Generalized Eigenvalue beamforming method
%  for isolating sound sources within a target Field of View (FOV).
%  
%  Author: Signal Processing Research Implementation
%  Method: MPDR Estimator + Matrix Integration + GEV Decomposition
%  ========================================================================

clear; close all; clc;

%% ========================================================================
%  CONFIGURATION FLAGS AND PARAMETERS
%  ========================================================================

% Toggle between synthetic and real data
USE_REAL_DATA = false;
REAL_DATA_FILE = 'array_input.wav';

% Audio parameters
fs = 16000;             % Sampling frequency (Hz)
duration = 5;           % Duration for synthetic signals (seconds)
c = 343;                % Speed of sound (m/s)

% Array geometry: 6-microphone Uniform Circular Array
num_mics = 6;
array_radius = 0.05;    % 5 cm radius

% Processing parameters
fft_size = 512;         % FFT size for STFT
hop_size = 128;         % Hop size (75% overlap)
freq_min = 300;         % Minimum frequency (Hz) - avoid low-freq noise
freq_max = 3400;        % Maximum frequency (Hz) - avoid spatial aliasing

% Beamforming parameters
target_direction = 0;           % Target source direction (degrees)
interference_direction = 90;    % Interference direction (degrees)
target_FOV = [-15, 15];         % Field of View for target (degrees)
angular_resolution = 20;        % Angular sampling for optimization (degrees)

% Regularization parameter for matrix inversion
reg_param = 1e-6;

%% ========================================================================
%  MICROPHONE ARRAY GEOMETRY
%  ========================================================================

fprintf('=== Audiovisual Zooming Simulation ===\n\n');
fprintf('Array Configuration:\n');
fprintf('  - %d microphones in Uniform Circular Array\n', num_mics);
fprintf('  - Radius: %.2f m\n', array_radius);

% Calculate microphone positions (UCA geometry)
mic_angles = (0:num_mics-1) * (360/num_mics);  % Degrees
mic_positions = zeros(num_mics, 2);
for m = 1:num_mics
    mic_positions(m, 1) = array_radius * cosd(mic_angles(m));  % x
    mic_positions(m, 2) = array_radius * sind(mic_angles(m));  % y
end

fprintf('  - Mic positions (x, y) in meters:\n');
for m = 1:num_mics
    fprintf('    Mic %d: (%.4f, %.4f)\n', m, mic_positions(m,1), mic_positions(m,2));
end

%% ========================================================================
%  DATA ACQUISITION / GENERATION
%  ========================================================================

if USE_REAL_DATA
    %% Load Real Data
    fprintf('\n[MODE] Loading REAL data from: %s\n', REAL_DATA_FILE);
    
    if ~exist(REAL_DATA_FILE, 'file')
        error('File not found: %s\nPlease record a 6-channel audio file.', REAL_DATA_FILE);
    end
    
    [mic_signals, fs_file] = audioread(REAL_DATA_FILE);
    
    % Resample if necessary
    if fs_file ~= fs
        fprintf('  Resampling from %d Hz to %d Hz...\n', fs_file, fs);
        mic_signals = resample(mic_signals, fs, fs_file);
    end
    
    % Verify channel count
    if size(mic_signals, 2) ~= num_mics
        error('Expected %d channels, got %d', num_mics, size(mic_signals, 2));
    end
    
    num_samples = size(mic_signals, 1);
    fprintf('  Loaded %d samples (%.2f seconds)\n', num_samples, num_samples/fs);
    
else
    %% Generate Synthetic Data
    fprintf('\n[MODE] Generating SYNTHETIC data\n');
    
    num_samples = duration * fs;
    t = (0:num_samples-1)' / fs;
    
    % ---- Generate Target Signal (Speech-like) at 0 degrees ----
    % Simulate speech with multiple harmonics and modulation
    f0 = 150;  % Fundamental frequency
    target_signal = zeros(num_samples, 1);
    
    % Add harmonics typical of speech
    for harmonic = 1:10
        freq = f0 * harmonic;
        if freq < freq_max
            amplitude = 1 / harmonic;  % Natural harmonic decay
            target_signal = target_signal + amplitude * sin(2*pi*freq*t);
        end
    end
    
    % Add amplitude modulation (syllable-like)
    modulation = 0.5 + 0.5 * sin(2*pi*4*t);  % 4 Hz modulation
    target_signal = target_signal .* modulation;
    target_signal = target_signal / max(abs(target_signal));
    
    % ---- Generate Interference Signal (Noise-like) at 90 degrees ----
    % Band-limited noise in the processing range
    interference_signal = randn(num_samples, 1);
    
    % Bandpass filter the interference
    [b_bp, a_bp] = butter(4, [freq_min, freq_max]/(fs/2), 'bandpass');
    interference_signal = filter(b_bp, a_bp, interference_signal);
    interference_signal = interference_signal / max(abs(interference_signal));
    
    % Scale interference relative to target (SNR control)
    interference_level = 0.8;  % Relative level
    interference_signal = interference_level * interference_signal;
    
    fprintf('  Target signal: Speech-like at %d degrees\n', target_direction);
    fprintf('  Interference: Band-limited noise at %d degrees\n', interference_direction);
    
    % ---- Apply Propagation Delays to Each Microphone ----
    mic_signals = zeros(num_samples, num_mics);
    
    for m = 1:num_mics
        % Calculate delay for target signal
        delay_target = compute_propagation_delay(...
            mic_positions(m, :), target_direction, c);
        
        % Calculate delay for interference signal
        delay_interference = compute_propagation_delay(...
            mic_positions(m, :), interference_direction, c);
        
        % Apply fractional delays using sinc interpolation
        target_delayed = apply_fractional_delay(target_signal, delay_target * fs);
        interference_delayed = apply_fractional_delay(interference_signal, delay_interference * fs);
        
        % Combine signals
        mic_signals(:, m) = target_delayed + interference_delayed;
    end
    
    % ---- Add Sensor Noise ----
    sensor_noise_level = 0.01;
    sensor_noise = sensor_noise_level * randn(num_samples, num_mics);
    mic_signals = mic_signals + sensor_noise;
    
    % Normalize
    mic_signals = mic_signals / max(abs(mic_signals(:)));
    
    fprintf('  Sensor noise level: %.1f%%\n', sensor_noise_level * 100);
    fprintf('  Total samples: %d (%.2f seconds)\n', num_samples, duration);
end

%% ========================================================================
%  SHORT-TIME FOURIER TRANSFORM (STFT)
%  ========================================================================

fprintf('\nPerforming STFT analysis...\n');

% Analysis window
window = hann(fft_size, 'periodic');

% Calculate number of frames
num_frames = floor((num_samples - fft_size) / hop_size) + 1;

% Frequency axis
freq_axis = (0:fft_size/2) * fs / fft_size;

% Find frequency bins within processing range
freq_bins = find(freq_axis >= freq_min & freq_axis <= freq_max);
num_freq_bins = length(freq_bins);

fprintf('  FFT size: %d\n', fft_size);
fprintf('  Hop size: %d (%.1f%% overlap)\n', hop_size, (1-hop_size/fft_size)*100);
fprintf('  Number of frames: %d\n', num_frames);
fprintf('  Processing %d frequency bins (%.0f-%.0f Hz)\n', ...
    num_freq_bins, freq_min, freq_max);

% Compute STFT for all channels
stft_data = zeros(fft_size/2+1, num_frames, num_mics);

for m = 1:num_mics
    for frame = 1:num_frames
        start_idx = (frame-1) * hop_size + 1;
        end_idx = start_idx + fft_size - 1;
        
        if end_idx <= num_samples
            segment = mic_signals(start_idx:end_idx, m) .* window;
            spectrum = fft(segment);
            stft_data(:, frame, m) = spectrum(1:fft_size/2+1);
        end
    end
end

%% ========================================================================
%  AUDIOVISUAL ZOOMING ALGORITHM
%  ========================================================================

fprintf('\n=== Audiovisual Zooming Processing ===\n');
fprintf('Target FOV: [%d, %d] degrees\n', target_FOV(1), target_FOV(2));
fprintf('Angular resolution: %d degrees\n', angular_resolution);

% Define scanning angles
scan_angles = -180:angular_resolution:180-angular_resolution;
num_angles = length(scan_angles);

% Identify angles inside and outside FOV
in_fov_mask = (scan_angles >= target_FOV(1)) & (scan_angles <= target_FOV(2));
out_fov_mask = ~in_fov_mask;

fprintf('  Angles in FOV: %d\n', sum(in_fov_mask));
fprintf('  Angles outside FOV: %d\n', sum(out_fov_mask));

% Pre-compute steering vectors for all angles and frequencies
fprintf('\nPre-computing steering vectors...\n');
steering_vectors = zeros(num_mics, num_angles, num_freq_bins);

for f_idx = 1:num_freq_bins
    freq = freq_axis(freq_bins(f_idx));
    wavelength = c / freq;
    
    for a_idx = 1:num_angles
        steering_vectors(:, a_idx, f_idx) = compute_steering_vector(...
            mic_positions, scan_angles(a_idx), wavelength);
    end
end

% Initialize output STFT
output_stft = zeros(fft_size/2+1, num_frames);

% Store beamformer weights for analysis
beamformer_weights = zeros(num_mics, num_freq_bins);

% Process each frequency bin
fprintf('Processing frequency bins...\n');
progress_step = floor(num_freq_bins / 10);

for f_idx = 1:num_freq_bins
    
    if mod(f_idx, progress_step) == 0
        fprintf('  Progress: %d%%\n', round(f_idx/num_freq_bins*100));
    end
    
    bin_idx = freq_bins(f_idx);
    freq = freq_axis(bin_idx);
    
    %% Step B: Covariance Estimation
    % Compute sample spectral matrix for current frequency bin
    X = squeeze(stft_data(bin_idx, :, :)).';  % num_mics x num_frames
    R_hat = (X * X') / num_frames;
    
    % Add regularization for numerical stability
    R_hat = R_hat + reg_param * eye(num_mics);
    
    % Compute inverse for MPDR
    R_inv = inv(R_hat);
    
    %% Step C: MPDR Power Map (Eq. 10)
    % P(theta) = [v_theta^H * R^-1 * v_theta]^-1
    power_map = zeros(1, num_angles);
    
    for a_idx = 1:num_angles
        v = steering_vectors(:, a_idx, f_idx);
        power_map(a_idx) = real(1 / (v' * R_inv * v));
    end
    
    % Ensure non-negative power
    power_map = max(power_map, 0);
    
    %% Step D: Matrix Integration (Eq. 11)
    % Signal matrix: integrate over FOV
    R_s = zeros(num_mics, num_mics);
    for a_idx = find(in_fov_mask)
        v = steering_vectors(:, a_idx, f_idx);
        R_s = R_s + power_map(a_idx) * (v * v');
    end
    
    % Noise matrix: integrate outside FOV
    R_n = zeros(num_mics, num_mics);
    for a_idx = find(out_fov_mask)
        v = steering_vectors(:, a_idx, f_idx);
        R_n = R_n + power_map(a_idx) * (v * v');
    end
    
    % Add regularization
    R_s = R_s + reg_param * eye(num_mics);
    R_n = R_n + reg_param * eye(num_mics);
    
    %% Step E: Generalized Eigenvalue Problem (Eq. 9)
    % Solve: R_s * w = lambda * R_n * w
    % Find eigenvector with largest eigenvalue
    
    try
        [V, D] = eig(R_s, R_n);
        eigenvalues = real(diag(D));
        [~, max_idx] = max(eigenvalues);
        w_opt = V(:, max_idx);
        
        % Normalize beamformer weights
        w_opt = w_opt / norm(w_opt);
        
    catch
        % Fallback to MPDR if GEV fails
        v_target = steering_vectors(:, find(scan_angles == 0, 1), f_idx);
        w_opt = R_inv * v_target / (v_target' * R_inv * v_target);
    end
    
    % Store weights for analysis
    beamformer_weights(:, f_idx) = w_opt;
    
    %% Apply Beamformer
    % y(f, t) = w^H * x(f, t)
    for frame = 1:num_frames
        x_frame = squeeze(stft_data(bin_idx, frame, :));
        output_stft(bin_idx, frame) = w_opt' * x_frame;
    end
end

% Copy unprocessed frequency bins (outside processing range)
for bin_idx = 1:fft_size/2+1
    if bin_idx < freq_bins(1) || bin_idx > freq_bins(end)
        % Use simple delay-and-sum for out-of-band frequencies
        output_stft(bin_idx, :) = mean(stft_data(bin_idx, :, :), 3);
    end
end

fprintf('  Processing complete!\n');

%% ========================================================================
%  INVERSE STFT - RECONSTRUCT TIME DOMAIN SIGNAL
%  ========================================================================

fprintf('\nReconstructing time-domain signal...\n');

% Synthesis window
synthesis_window = hann(fft_size, 'periodic');

% Output signal buffer
output_signal = zeros(num_samples, 1);
window_sum = zeros(num_samples, 1);

for frame = 1:num_frames
    start_idx = (frame-1) * hop_size + 1;
    end_idx = start_idx + fft_size - 1;
    
    if end_idx <= num_samples
        % Create full spectrum (conjugate symmetric)
        full_spectrum = zeros(fft_size, 1);
        full_spectrum(1:fft_size/2+1) = output_stft(:, frame);
        full_spectrum(fft_size/2+2:end) = conj(flipud(output_stft(2:end-1, frame)));
        
        % IFFT
        frame_signal = real(ifft(full_spectrum));
        
        % Overlap-add with synthesis window
        output_signal(start_idx:end_idx) = output_signal(start_idx:end_idx) + ...
            frame_signal .* synthesis_window;
        window_sum(start_idx:end_idx) = window_sum(start_idx:end_idx) + ...
            synthesis_window.^2;
    end
end

% Normalize by window sum
valid_idx = window_sum > 1e-8;
output_signal(valid_idx) = output_signal(valid_idx) ./ window_sum(valid_idx);

% Normalize output
output_signal = output_signal / max(abs(output_signal));

% Reference signal (first microphone)
reference_signal = mic_signals(:, 1);
reference_signal = reference_signal / max(abs(reference_signal));

fprintf('  Output signal reconstructed.\n');

%% ========================================================================
%  BEAM PATTERN ANALYSIS
%  ========================================================================

fprintf('\nComputing beam pattern at 1000 Hz...\n');

% Find frequency bin closest to 1000 Hz
[~, analysis_freq_idx] = min(abs(freq_axis(freq_bins) - 1000));
analysis_freq = freq_axis(freq_bins(analysis_freq_idx));
fprintf('  Analysis frequency: %.1f Hz\n', analysis_freq);

% Get beamformer weights at this frequency
w_analysis = beamformer_weights(:, analysis_freq_idx);

% Compute beam pattern with fine angular resolution
beam_angles = -180:1:179;
beam_pattern = zeros(1, length(beam_angles));
wavelength_analysis = c / analysis_freq;

for a_idx = 1:length(beam_angles)
    v = compute_steering_vector(mic_positions, beam_angles(a_idx), wavelength_analysis);
    beam_pattern(a_idx) = abs(w_analysis' * v)^2;
end

% Normalize beam pattern
beam_pattern_dB = 10 * log10(beam_pattern / max(beam_pattern) + 1e-10);

% Find gains at target and interference directions
[~, target_idx] = min(abs(beam_angles - target_direction));
[~, interf_idx] = min(abs(beam_angles - interference_direction));

target_gain_dB = beam_pattern_dB(target_idx);
interf_gain_dB = beam_pattern_dB(interf_idx);

fprintf('  Gain at target (%d deg): %.2f dB\n', target_direction, target_gain_dB);
fprintf('  Gain at interference (%d deg): %.2f dB\n', interference_direction, interf_gain_dB);
fprintf('  Suppression: %.2f dB\n', target_gain_dB - interf_gain_dB);

%% ========================================================================
%  VISUALIZATION
%  ========================================================================

fprintf('\nGenerating visualizations...\n');

figure('Position', [50, 50, 1400, 900], 'Color', 'w');

% ---- Subplot 1: Array Geometry ----
subplot(2, 3, 1);
plot(mic_positions(:, 1)*100, mic_positions(:, 2)*100, 'ko', ...
    'MarkerSize', 12, 'MarkerFaceColor', [0.2, 0.6, 0.9], 'LineWidth', 2);
hold on;
theta_circle = linspace(0, 2*pi, 100);
plot(array_radius*100*cos(theta_circle), array_radius*100*sin(theta_circle), ...
    'b--', 'LineWidth', 1);

% Draw target direction
arrow_len = array_radius * 100 * 1.5;
quiver(0, 0, arrow_len*cosd(target_direction), arrow_len*sind(target_direction), 0, ...
    'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, arrow_len*cosd(interference_direction), arrow_len*sind(interference_direction), 0, ...
    'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Draw FOV
fov_arc_angles = linspace(target_FOV(1), target_FOV(2), 50);
fov_radius = array_radius * 100 * 1.3;
fill([0, fov_radius*cosd(fov_arc_angles), 0], ...
     [0, fov_radius*sind(fov_arc_angles), 0], ...
     'g', 'FaceAlpha', 0.2, 'EdgeColor', 'g', 'LineWidth', 1.5);

grid on; axis equal;
xlabel('X (cm)'); ylabel('Y (cm)');
title('6-Mic Uniform Circular Array');
legend('Microphones', 'Array', 'Target', 'Interference', 'FOV', ...
    'Location', 'best');
xlim([-10, 10]); ylim([-10, 10]);

% ---- Subplot 2: Input Spectrogram ----
subplot(2, 3, 2);
[~, freq_spec, time_spec, psd_in] = spectrogram(reference_signal, ...
    hann(256), 192, 256, fs, 'yaxis');
imagesc(time_spec, freq_spec, 10*log10(abs(psd_in) + 1e-10));
axis xy; colormap(gca, 'jet'); colorbar;
ylim([0, 4000]);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Input Spectrogram (Mic 1 - Noisy)');
caxis([-60, 0]);

% ---- Subplot 3: Output Spectrogram ----
subplot(2, 3, 3);
[~, ~, ~, psd_out] = spectrogram(output_signal, ...
    hann(256), 192, 256, fs, 'yaxis');
imagesc(time_spec, freq_spec, 10*log10(abs(psd_out) + 1e-10));
axis xy; colormap(gca, 'jet'); colorbar;
ylim([0, 4000]);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Output Spectrogram (Zoomed - Enhanced)');
caxis([-60, 0]);

% ---- Subplot 4: Time Domain Comparison ----
subplot(2, 3, 4);
t_plot = (0:num_samples-1) / fs;
plot(t_plot, reference_signal, 'b', 'LineWidth', 0.5);
hold on;
plot(t_plot, output_signal, 'r', 'LineWidth', 0.5);
xlabel('Time (s)'); ylabel('Amplitude');
title('Time Domain: Input vs Output');
legend('Input (Noisy)', 'Output (Zoomed)', 'Location', 'best');
grid on;
xlim([0, min(duration, num_samples/fs)]);

% ---- Subplot 5: Beam Pattern (Cartesian) ----
subplot(2, 3, 5);
plot(beam_angles, beam_pattern_dB, 'b-', 'LineWidth', 2);
hold on;
xline(target_direction, 'g--', 'LineWidth', 2);
xline(interference_direction, 'r--', 'LineWidth', 2);

% Shade FOV region
y_lim = ylim;
fill([target_FOV(1), target_FOV(2), target_FOV(2), target_FOV(1)], ...
     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
     'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

xlabel('Angle (degrees)'); ylabel('Gain (dB)');
title(sprintf('Beam Pattern at %.0f Hz', analysis_freq));
legend('Beam Pattern', 'Target', 'Interference', 'FOV', 'Location', 'best');
grid on;
xlim([-180, 180]);
ylim([-40, 5]);

% ---- Subplot 6: Beam Pattern (Polar) ----
subplot(2, 3, 6);
% Convert to polar
beam_pattern_linear = 10.^(beam_pattern_dB/20);
beam_pattern_linear = max(beam_pattern_linear, 0.01);  % Clip for visualization

polarplot(deg2rad(beam_angles), beam_pattern_linear, 'b-', 'LineWidth', 2);
hold on;
polarplot(deg2rad(target_direction), beam_pattern_linear(target_idx), ...
    'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
polarplot(deg2rad(interference_direction), beam_pattern_linear(interf_idx), ...
    'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

% Draw FOV arc
fov_angles_rad = deg2rad(linspace(target_FOV(1), target_FOV(2), 50));
polarplot(fov_angles_rad, ones(size(fov_angles_rad)), 'g-', 'LineWidth', 3);

title(sprintf('Polar Beam Pattern at %.0f Hz', analysis_freq));
legend('Pattern', 'Target', 'Interference', 'FOV', 'Location', 'best');
rlim([0, 1.2]);

sgtitle('Audiovisual Zooming: Generalized Eigenvalue Beamformer', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  ADDITIONAL ANALYSIS FIGURE
%  ========================================================================

figure('Position', [100, 100, 1200, 500], 'Color', 'w');

% ---- MPDR Spatial Power Map ----
subplot(1, 2, 1);

% Compute average MPDR power map across all frequency bins
avg_power_map = zeros(1, num_angles);
fine_scan_angles = -180:5:175;
num_fine_angles = length(fine_scan_angles);
power_map_freq = zeros(num_freq_bins, num_fine_angles);

for f_idx = 1:num_freq_bins
    bin_idx = freq_bins(f_idx);
    freq = freq_axis(bin_idx);
    wavelength = c / freq;
    
    X = squeeze(stft_data(bin_idx, :, :)).';
    R_hat = (X * X') / num_frames + reg_param * eye(num_mics);
    R_inv = inv(R_hat);
    
    for a_idx = 1:num_fine_angles
        v = compute_steering_vector(mic_positions, fine_scan_angles(a_idx), wavelength);
        power_map_freq(f_idx, a_idx) = real(1 / (v' * R_inv * v));
    end
end

% Normalize each frequency bin
for f_idx = 1:num_freq_bins
    power_map_freq(f_idx, :) = power_map_freq(f_idx, :) / max(power_map_freq(f_idx, :));
end

imagesc(fine_scan_angles, freq_axis(freq_bins), 10*log10(power_map_freq + 1e-10));
axis xy; colormap(gca, 'hot'); colorbar;
xlabel('Direction of Arrival (degrees)');
ylabel('Frequency (Hz)');
title('MPDR Spatial Power Spectrum');
hold on;
xline(target_direction, 'g--', 'LineWidth', 2);
xline(interference_direction, 'c--', 'LineWidth', 2);
xline(target_FOV(1), 'w--', 'LineWidth', 1);
xline(target_FOV(2), 'w--', 'LineWidth', 1);
legend('Target', 'Interference', 'FOV Bounds', 'Location', 'best');
caxis([-20, 0]);

% ---- Frequency Response Comparison ----
subplot(1, 2, 2);

% Compute average power spectrum
nfft_analysis = 1024;
[pxx_in, f_pxx] = pwelch(reference_signal, hann(nfft_analysis), nfft_analysis/2, nfft_analysis, fs);
[pxx_out, ~] = pwelch(output_signal, hann(nfft_analysis), nfft_analysis/2, nfft_analysis, fs);

plot(f_pxx, 10*log10(pxx_in), 'b-', 'LineWidth', 1.5);
hold on;
plot(f_pxx, 10*log10(pxx_out), 'r-', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Power Spectrum: Input vs Output');
legend('Input (Noisy)', 'Output (Zoomed)', 'Location', 'best');
grid on;
xlim([0, 4000]);

% Add processing range indicator
y_lim = ylim;
fill([freq_min, freq_max, freq_max, freq_min], ...
     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
     'g', 'FaceAlpha', 0.1, 'EdgeColor', 'g', 'LineStyle', '--');

sgtitle('Spatial Analysis Results', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  AUDIO PLAYBACK
%  ========================================================================

fprintf('\n=== Audio Playback ===\n');
fprintf('Playing INPUT signal (noisy)...\n');
soundsc(reference_signal, fs);
pause(duration + 0.5);

fprintf('Playing OUTPUT signal (zoomed/enhanced)...\n');
soundsc(output_signal, fs);
pause(duration + 0.5);

%% ========================================================================
%  SAVE RESULTS
%  ========================================================================

fprintf('\n=== Saving Results ===\n');

% Save output audio
output_filename = 'zoomed_output.wav';
audiowrite(output_filename, output_signal, fs);
fprintf('  Output audio saved to: %s\n', output_filename);

% Save input reference
input_filename = 'input_reference.wav';
audiowrite(input_filename, reference_signal, fs);
fprintf('  Input reference saved to: %s\n', input_filename);

%% ========================================================================
%  PERFORMANCE METRICS
%  ========================================================================

fprintf('\n=== Performance Summary ===\n');
fprintf('  Target Direction: %d degrees\n', target_direction);
fprintf('  Interference Direction: %d degrees\n', interference_direction);
fprintf('  Target FOV: [%d, %d] degrees\n', target_FOV(1), target_FOV(2));
fprintf('  Beam Pattern Analysis (at %.0f Hz):\n', analysis_freq);
fprintf('    - Gain at target: %.2f dB\n', target_gain_dB);
fprintf('    - Gain at interference: %.2f dB\n', interf_gain_dB);
fprintf('    - Interference suppression: %.2f dB\n', target_gain_dB - interf_gain_dB);

% Estimate SNR improvement (approximate)
if ~USE_REAL_DATA
    % For synthetic data, we can compute SNR
    % This is approximate since we don't have perfect separation
    
    % Energy in target direction band (estimated)
    target_energy_in = sum(reference_signal.^2);
    target_energy_out = sum(output_signal.^2);
    
    fprintf('  Energy Analysis:\n');
    fprintf('    - Input energy: %.4f\n', target_energy_in);
    fprintf('    - Output energy: %.4f\n', target_energy_out);
end

fprintf('\n=== Simulation Complete ===\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function delay = compute_propagation_delay(mic_pos, angle_deg, c)
    % Compute propagation delay for a plane wave arriving from angle_deg
    % Input:
    %   mic_pos: [x, y] position of microphone (m)
    %   angle_deg: Direction of arrival (degrees, 0 = +x axis)
    %   c: Speed of sound (m/s)
    % Output:
    %   delay: Time delay (seconds)
    
    % Unit vector pointing toward source
    ux = cosd(angle_deg);
    uy = sind(angle_deg);
    
    % Path difference (negative because wave comes FROM this direction)
    path_diff = -(mic_pos(1) * ux + mic_pos(2) * uy);
    
    % Time delay
    delay = path_diff / c;
end

function v = compute_steering_vector(mic_positions, angle_deg, wavelength)
    % Compute steering vector for given direction and frequency
    % Input:
    %   mic_positions: num_mics x 2 matrix of [x, y] positions
    %   angle_deg: Direction of arrival (degrees)
    %   wavelength: Wavelength (m)
    % Output:
    %   v: Steering vector (num_mics x 1)
    
    num_mics = size(mic_positions, 1);
    k = 2 * pi / wavelength;  % Wave number
    
    % Unit vector pointing toward source
    ux = cosd(angle_deg);
    uy = sind(angle_deg);
    
    % Phase delays
    v = zeros(num_mics, 1);
    for m = 1:num_mics
        % Path difference from reference (center of array)
        path_diff = mic_positions(m, 1) * ux + mic_positions(m, 2) * uy;
        v(m) = exp(-1j * k * path_diff);
    end
end

function y = apply_fractional_delay(x, delay_samples)
    % Apply fractional delay using sinc interpolation
    % Input:
    %   x: Input signal
    %   delay_samples: Delay in samples (can be fractional)
    % Output:
    %   y: Delayed signal
    
    n = length(x);
    y = zeros(n, 1);
    
    % Integer and fractional parts
    int_delay = floor(delay_samples);
    frac_delay = delay_samples - int_delay;
    
    if abs(frac_delay) < 1e-10
        % Integer delay only
        if int_delay >= 0
            y(int_delay+1:end) = x(1:end-int_delay);
        else
            y(1:end+int_delay) = x(-int_delay+1:end);
        end
    else
        % Sinc interpolation for fractional delay
        filter_len = 31;  % Filter length (odd)
        half_len = (filter_len - 1) / 2;
        
        % Design fractional delay filter
        n_filt = -half_len:half_len;
        h = sinc(n_filt - frac_delay) .* hamming(filter_len)';
        h = h / sum(h);
        
        % Apply filter
        x_filtered = conv(x, h, 'same');
        
        % Apply integer delay
        if int_delay >= 0
            y(int_delay+1:end) = x_filtered(1:end-int_delay);
        else
            y(1:end+int_delay) = x_filtered(-int_delay+1:end);
        end
    end
end

