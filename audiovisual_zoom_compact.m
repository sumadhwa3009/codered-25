%% ========================================================================
%  AUDIOVISUAL ZOOMING - GEV Beamformer
%  "What You See Is What You Hear" (Nair et al., 2019)
%  ========================================================================
%  Algorithm: MPDR Power Estimation → Matrix Integration → GEV Solution
%  Requires: Signal Processing Toolbox
%  ========================================================================

clear; close all; clc;
fprintf('=== Audiovisual Zooming Simulation ===\n\n');

%% ======================= CONFIGURATION ==================================

% Audio settings
fs          = 16000;        % Sampling frequency (Hz)
duration    = 3;            % Signal duration (seconds)
c           = 343;          % Speed of sound (m/s)

% Array geometry: 6-mic Uniform Circular Array
num_mics    = 6;
radius      = 0.05;         % 5 cm radius

% Processing settings
fft_len     = 512;          % FFT length
hop_size    = 128;          % Hop size (75% overlap)
freq_min    = 300;          % Min frequency (Hz)
freq_max    = 3400;         % Max frequency (Hz)

% Beamforming settings
target_dir  = 0;            % Target direction (degrees)
interf_dir  = 90;           % Interference direction (degrees)
fov         = [-15, 15];    % Field of View (degrees)
scan_step   = 20;           % Angular resolution for scanning

% Regularization for numerical stability
reg = 1e-6;

%% ======================= ARRAY GEOMETRY =================================

mic_angles = (0:num_mics-1) * (360/num_mics);  % 0, 60, 120, 180, 240, 300
mic_pos = radius * [cosd(mic_angles)', sind(mic_angles)'];

fprintf('Array: %d mics, %.0f cm radius\n', num_mics, radius*100);
fprintf('Target: %d°  |  Interference: %d°  |  FOV: [%d°, %d°]\n\n', ...
    target_dir, interf_dir, fov(1), fov(2));

%% ======================= GENERATE SIGNALS ===============================

num_samples = duration * fs;
t = (0:num_samples-1)' / fs;

% Target: Speech-like harmonic signal with AM modulation
harmonics = 200:200:1600;  % 200, 400, 600, ... 1600 Hz
target_signal = zeros(num_samples, 1);
for i = 1:length(harmonics)
    target_signal = target_signal + sin(2*pi*harmonics(i)*t) / i;
end
target_signal = target_signal .* (0.5 + 0.5*sin(2*pi*3*t));  % 3 Hz AM
target_signal = target_signal / max(abs(target_signal));

% Interference: Band-limited noise
interf_signal = bandpass(randn(num_samples, 1), [freq_min freq_max], fs);
interf_signal = 0.7 * interf_signal / max(abs(interf_signal));

fprintf('Generated: Target (speech-like) + Interference (noise)\n');

%% ======================= APPLY PROPAGATION DELAYS =======================

mic_signals = zeros(num_samples, num_mics);
freq_vec = (0:num_samples-1)' * fs / num_samples;
freq_vec(freq_vec > fs/2) = freq_vec(freq_vec > fs/2) - fs;

for m = 1:num_mics
    % Time delay = (mic position . direction unit vector) / speed
    tau_target = (mic_pos(m,1)*cosd(target_dir) + mic_pos(m,2)*sind(target_dir)) / c;
    tau_interf = (mic_pos(m,1)*cosd(interf_dir) + mic_pos(m,2)*sind(interf_dir)) / c;
    
    % Apply delays in frequency domain (accurate fractional delay)
    target_delayed = real(ifft(fft(target_signal) .* exp(-1j*2*pi*freq_vec*tau_target)));
    interf_delayed = real(ifft(fft(interf_signal) .* exp(-1j*2*pi*freq_vec*tau_interf)));
    
    % Mix signals + sensor noise
    mic_signals(:, m) = target_delayed + interf_delayed + 0.02*randn(num_samples, 1);
end

mic_signals = mic_signals / max(abs(mic_signals(:)));
fprintf('Applied propagation delays to %d microphones\n', num_mics);

%% ======================= STFT ANALYSIS ==================================

[S, freq_axis, time_axis] = stft(mic_signals, fs, ...
    'Window', hann(fft_len, 'periodic'), ...
    'OverlapLength', fft_len - hop_size, ...
    'FFTLength', fft_len);

[num_freqs, num_frames, ~] = size(S);
freq_bins = find(freq_axis >= freq_min & freq_axis <= freq_max);

fprintf('STFT: %d frames, processing %d frequency bins\n', num_frames, length(freq_bins));

% Initialize output with channel average
output_stft = mean(S, 3);

%% ======================= GEV BEAMFORMING ================================

scan_angles = -180:scan_step:180-scan_step;
num_angles = length(scan_angles);

fprintf('Running GEV beamformer...\n');

for idx = 1:length(freq_bins)
    bin = freq_bins(idx);
    freq = freq_axis(bin);
    k = 2*pi * freq / c;  % Wavenumber
    
    % --- Extract multichannel data for this frequency ---
    X = squeeze(S(bin, :, :)).';  % [num_mics x num_frames]
    
    % --- Step 1: Covariance Matrix ---
    R = (X * X') / num_frames + reg * eye(num_mics);
    R_inv = inv(R);
    
    % --- Step 2: Steering Vectors ---
    % V(m, a) = exp(-j * k * (x_m * cos(θ) + y_m * sin(θ)))
    V = exp(-1j * k * (mic_pos(:,1) * cosd(scan_angles) + mic_pos(:,2) * sind(scan_angles)));
    
    % --- Step 3: MPDR Power Map ---
    % P(θ) = 1 / (v'R⁻¹v)
    P = real(1 ./ sum(conj(V) .* (R_inv * V), 1));
    P = max(P, 0);  % Ensure non-negative
    
    % --- Step 4: Matrix Integration ---
    in_fov = (scan_angles >= fov(1)) & (scan_angles <= fov(2));
    
    % Signal matrix: sum over FOV
    V_sig = V(:, in_fov);
    P_sig = P(in_fov);
    R_signal = V_sig * diag(P_sig) * V_sig' + reg * eye(num_mics);
    
    % Noise matrix: sum outside FOV
    V_noise = V(:, ~in_fov);
    P_noise = P(~in_fov);
    R_noise = V_noise * diag(P_noise) * V_noise' + reg * eye(num_mics);
    
    % --- Step 5: Generalized Eigenvalue Problem ---
    % Solve: R_signal * w = λ * R_noise * w
    [eigvecs, eigvals] = eig(R_signal, R_noise);
    [~, max_idx] = max(real(diag(eigvals)));
    w = eigvecs(:, max_idx);
    w = w / norm(w);
    
    % --- Apply beamformer ---
    output_stft(bin, :) = w' * X;
end

fprintf('Beamforming complete.\n');

%% ======================= RECONSTRUCT OUTPUT =============================

output_signal = istft(output_stft, fs, ...
    'Window', hann(fft_len, 'periodic'), ...
    'OverlapLength', fft_len - hop_size, ...
    'FFTLength', fft_len);

output_signal = real(output_signal);
output_signal = output_signal(1:min(num_samples, length(output_signal)));
output_signal = output_signal / max(abs(output_signal));

% Reference: first microphone
input_signal = mic_signals(:, 1) / max(abs(mic_signals(:, 1)));

%% ======================= BEAM PATTERN AT 1 kHz ==========================

analysis_freq = 1000;
k_analysis = 2*pi * analysis_freq / c;
pattern_angles = -180:1:179;

% Find frequency bin
[~, bin_1k] = min(abs(freq_axis - analysis_freq));

% Get data at 1 kHz
X_1k = squeeze(S(bin_1k, :, :)).';
R_1k = (X_1k * X_1k') / num_frames + reg * eye(num_mics);
R_1k_inv = inv(R_1k);

% Steering vectors (fine resolution)
V_fine = exp(-1j * k_analysis * (mic_pos(:,1) * cosd(pattern_angles) + mic_pos(:,2) * sind(pattern_angles)));

% MPDR power
P_fine = real(1 ./ sum(conj(V_fine) .* (R_1k_inv * V_fine), 1));
P_fine = max(P_fine, 0);

% GEV weights at 1 kHz
V_scan = exp(-1j * k_analysis * (mic_pos(:,1) * cosd(scan_angles) + mic_pos(:,2) * sind(scan_angles)));
P_scan = real(1 ./ sum(conj(V_scan) .* (R_1k_inv * V_scan), 1));
P_scan = max(P_scan, 0);

in_fov_scan = (scan_angles >= fov(1)) & (scan_angles <= fov(2));
Rs_1k = V_scan(:,in_fov_scan) * diag(P_scan(in_fov_scan)) * V_scan(:,in_fov_scan)' + reg*eye(num_mics);
Rn_1k = V_scan(:,~in_fov_scan) * diag(P_scan(~in_fov_scan)) * V_scan(:,~in_fov_scan)' + reg*eye(num_mics);

[V_gev, D_gev] = eig(Rs_1k, Rn_1k);
[~, max_i] = max(real(diag(D_gev)));
w_gev = V_gev(:, max_i);
w_gev = w_gev / norm(w_gev);

% Compute beam pattern
beam_pattern = abs(w_gev' * V_fine).^2;
beam_pattern_dB = 10 * log10(beam_pattern / max(beam_pattern) + 1e-10);

% Get gains
gain_target = beam_pattern_dB(pattern_angles == target_dir);
gain_interf = beam_pattern_dB(pattern_angles == interf_dir);
suppression = gain_target - gain_interf;

fprintf('\n=== RESULTS ===\n');
fprintf('Gain at target (%d°):      %.1f dB\n', target_dir, gain_target);
fprintf('Gain at interference (%d°): %.1f dB\n', interf_dir, gain_interf);
fprintf('Suppression:                %.1f dB\n', suppression);

%% ======================= VISUALIZATION ==================================

figure('Position', [50, 50, 1300, 600], 'Color', 'w');

% Input spectrogram
subplot(2, 3, 1);
spectrogram(input_signal, hann(256), 192, 256, fs, 'yaxis');
title('Input (Noisy)');
ylim([0 4]);
colorbar;

% Output spectrogram
subplot(2, 3, 2);
spectrogram(output_signal, hann(256), 192, 256, fs, 'yaxis');
title('Output (Zoomed)');
ylim([0 4]);
colorbar;

% Waveforms
subplot(2, 3, 3);
plot(t, input_signal, 'b', 'LineWidth', 0.5);
hold on;
plot(t(1:length(output_signal)), output_signal, 'r', 'LineWidth', 0.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Waveform Comparison');
legend('Input', 'Output', 'Location', 'best');
grid on;
xlim([0 duration]);

% Beam pattern (Cartesian)
subplot(2, 3, 4);
plot(pattern_angles, beam_pattern_dB, 'b-', 'LineWidth', 2);
hold on;
xline(target_dir, 'g--', 'LineWidth', 2);
xline(interf_dir, 'r--', 'LineWidth', 2);
fill([fov(1), fov(2), fov(2), fov(1)], [-50, -50, 5, 5], ...
    'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
xlabel('Angle (degrees)');
ylabel('Gain (dB)');
title(sprintf('Beam Pattern @ %d Hz', analysis_freq));
legend('Pattern', 'Target', 'Interference', 'FOV', 'Location', 'best');
xlim([-180 180]);
ylim([-40 5]);
grid on;

% Beam pattern (Polar)
subplot(2, 3, 5);
beam_linear = max(10.^(beam_pattern_dB/20), 0.01);
polarplot(deg2rad(pattern_angles), beam_linear, 'b-', 'LineWidth', 2);
hold on;
polarplot(deg2rad(target_dir), 1, 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
polarplot(deg2rad(interf_dir), beam_linear(pattern_angles == interf_dir), ...
    'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
title('Polar Beam Pattern');
rlim([0 1.2]);

% Suppression bar chart
subplot(2, 3, 6);
bar([1, 2], [gain_target, gain_interf], 0.6);
set(gca, 'XTickLabel', {sprintf('Target %d°', target_dir), sprintf('Interf %d°', interf_dir)});
ylabel('Gain (dB)');
title(sprintf('Suppression: %.1f dB', suppression));
grid on;

sgtitle('Audiovisual Zooming - GEV Beamformer', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================= SAVE & PLAY AUDIO ==============================

audiowrite('output_zoomed.wav', output_signal, fs);
audiowrite('input_noisy.wav', input_signal, fs);
fprintf('\nSaved: input_noisy.wav, output_zoomed.wav\n');

fprintf('\nPlaying INPUT (noisy)...\n');
soundsc(input_signal, fs);
pause(duration + 0.5);

fprintf('Playing OUTPUT (zoomed)...\n');
soundsc(output_signal, fs);

fprintf('\n=== Done ===\n');
