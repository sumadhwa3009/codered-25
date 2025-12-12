%% ========================================================================
%  DEMO: Generalized Eigenvalue Beamformer
%  Simplified standalone demonstration of the Audiovisual Zooming algorithm
%  ========================================================================
%  This script demonstrates the core concepts in a minimal implementation.
%  For the full implementation, use audiovisual_zooming_simulation.m
%  ========================================================================

clear; close all; clc;

fprintf('=== GEV Beamformer Demo ===\n\n');

%% Setup Array and Sources
fs = 16000;             % Sampling rate
c = 343;                % Speed of sound
duration = 3;           % Signal duration
num_samples = fs * duration;
t = (0:num_samples-1)' / fs;

% 6-mic Uniform Circular Array (5cm radius)
num_mics = 6;
radius = 0.05;
mic_angles_deg = (0:num_mics-1) * 60;
mic_pos = radius * [cosd(mic_angles_deg)', sind(mic_angles_deg)'];

% Source directions
target_angle = 0;       % Target at 0 degrees
interf_angle = 90;      % Interference at 90 degrees

fprintf('Array: %d microphones, %.0f cm radius\n', num_mics, radius*100);
fprintf('Target: %d°, Interference: %d°\n', target_angle, interf_angle);

%% Generate Test Signals
% Target: Multi-harmonic speech-like signal
f0 = 200;  % Fundamental
target = zeros(num_samples, 1);
for h = 1:8
    target = target + (1/h) * sin(2*pi*f0*h*t);
end
target = target .* (0.5 + 0.5*sin(2*pi*3*t));  % AM modulation
target = target / max(abs(target));

% Interference: Band-limited noise
[b, a] = butter(4, [300, 3400]/(fs/2), 'bandpass');
interf = filter(b, a, randn(num_samples, 1));
interf = 0.7 * interf / max(abs(interf));

%% Apply Propagation Delays
mic_signals = zeros(num_samples, num_mics);

for m = 1:num_mics
    % Target delay
    tau_t = (mic_pos(m,1)*cosd(target_angle) + mic_pos(m,2)*sind(target_angle)) / c;
    % Interference delay
    tau_i = (mic_pos(m,1)*cosd(interf_angle) + mic_pos(m,2)*sind(interf_angle)) / c;
    
    % Apply phase shifts in frequency domain (accurate fractional delay)
    Target_fft = fft(target);
    Interf_fft = fft(interf);
    
    freq = (0:num_samples-1)' * fs / num_samples;
    freq(freq > fs/2) = freq(freq > fs/2) - fs;  % Negative frequencies
    
    Target_delayed = ifft(Target_fft .* exp(-1j*2*pi*freq*tau_t));
    Interf_delayed = ifft(Interf_fft .* exp(-1j*2*pi*freq*tau_i));
    
    mic_signals(:, m) = real(Target_delayed + Interf_delayed) + 0.02*randn(num_samples, 1);
end

mic_signals = mic_signals / max(abs(mic_signals(:)));

%% STFT Processing
fft_size = 512;
hop_size = 128;
window = hann(fft_size, 'periodic');
num_frames = floor((num_samples - fft_size) / hop_size) + 1;
freq_axis = (0:fft_size/2) * fs / fft_size;

% Find frequency bins in processing range
freq_bins = find(freq_axis >= 300 & freq_axis <= 3400);

fprintf('\nProcessing %d frequency bins...\n', length(freq_bins));

% Compute STFT
stft_all = zeros(fft_size/2+1, num_frames, num_mics);
for m = 1:num_mics
    for fr = 1:num_frames
        idx = (fr-1)*hop_size + 1;
        seg = mic_signals(idx:idx+fft_size-1, m) .* window;
        spec = fft(seg);
        stft_all(:, fr, m) = spec(1:fft_size/2+1);
    end
end

% Output STFT
output_stft = zeros(fft_size/2+1, num_frames);

% Scanning angles (coarse for speed)
scan_angles = -180:20:160;
FOV = [-20, 20];  % Target FOV

%% === CORE ALGORITHM ===
for f_idx = 1:length(freq_bins)
    bin = freq_bins(f_idx);
    freq = freq_axis(bin);
    lambda = c / freq;
    k = 2*pi / lambda;
    
    % --- Step B: Covariance Matrix ---
    X = squeeze(stft_all(bin, :, :)).';  % [num_mics x num_frames]
    R = (X * X') / num_frames + 1e-6*eye(num_mics);
    R_inv = inv(R);
    
    % --- Pre-compute steering vectors ---
    V = zeros(num_mics, length(scan_angles));
    for a = 1:length(scan_angles)
        for m = 1:num_mics
            path = mic_pos(m,1)*cosd(scan_angles(a)) + mic_pos(m,2)*sind(scan_angles(a));
            V(m, a) = exp(-1j * k * path);
        end
    end
    
    % --- Step C: MPDR Power Map ---
    P = zeros(1, length(scan_angles));
    for a = 1:length(scan_angles)
        v = V(:, a);
        P(a) = real(1 / (v' * R_inv * v));
    end
    P = max(P, 0);
    
    % --- Step D: Matrix Integration ---
    in_fov = (scan_angles >= FOV(1)) & (scan_angles <= FOV(2));
    
    R_s = zeros(num_mics);  % Signal matrix
    R_n = zeros(num_mics);  % Noise matrix
    
    for a = 1:length(scan_angles)
        v = V(:, a);
        outer = P(a) * (v * v');
        if in_fov(a)
            R_s = R_s + outer;
        else
            R_n = R_n + outer;
        end
    end
    
    R_s = R_s + 1e-6*eye(num_mics);
    R_n = R_n + 1e-6*eye(num_mics);
    
    % --- Step E: GEV Solution ---
    [Eigvecs, Eigvals] = eig(R_s, R_n);
    [~, max_idx] = max(real(diag(Eigvals)));
    w = Eigvecs(:, max_idx);
    w = w / norm(w);
    
    % --- Apply Beamformer ---
    output_stft(bin, :) = w' * X;
end

% Passthrough for out-of-band frequencies
for bin = 1:fft_size/2+1
    if bin < freq_bins(1) || bin > freq_bins(end)
        output_stft(bin, :) = mean(stft_all(bin, :, :), 3);
    end
end

%% Inverse STFT
output = zeros(num_samples, 1);
win_sum = zeros(num_samples, 1);

for fr = 1:num_frames
    idx = (fr-1)*hop_size + 1;
    
    full_spec = zeros(fft_size, 1);
    full_spec(1:fft_size/2+1) = output_stft(:, fr);
    full_spec(fft_size/2+2:end) = conj(flipud(output_stft(2:end-1, fr)));
    
    frame_sig = real(ifft(full_spec)) .* window;
    
    output(idx:idx+fft_size-1) = output(idx:idx+fft_size-1) + frame_sig;
    win_sum(idx:idx+fft_size-1) = win_sum(idx:idx+fft_size-1) + window.^2;
end

valid = win_sum > 1e-8;
output(valid) = output(valid) ./ win_sum(valid);
output = output / max(abs(output));

ref_input = mic_signals(:, 1) / max(abs(mic_signals(:, 1)));

%% Compute Beam Pattern at 1000 Hz
[~, bin_1k] = min(abs(freq_axis - 1000));
X_1k = squeeze(stft_all(bin_1k, :, :)).';
R_1k = (X_1k * X_1k') / num_frames + 1e-6*eye(num_mics);
R_1k_inv = inv(R_1k);

lambda_1k = c / 1000;
k_1k = 2*pi / lambda_1k;

% Fine resolution for plotting
plot_angles = -180:1:179;
V_plot = zeros(num_mics, length(plot_angles));
for a = 1:length(plot_angles)
    for m = 1:num_mics
        path = mic_pos(m,1)*cosd(plot_angles(a)) + mic_pos(m,2)*sind(plot_angles(a));
        V_plot(m, a) = exp(-1j * k_1k * path);
    end
end

% MPDR for beam pattern
P_fine = zeros(1, length(plot_angles));
for a = 1:length(plot_angles)
    v = V_plot(:, a);
    P_fine(a) = real(1 / (v' * R_1k_inv * v));
end

% GEV weights at 1000 Hz
in_fov_coarse = (scan_angles >= FOV(1)) & (scan_angles <= FOV(2));
V_coarse = zeros(num_mics, length(scan_angles));
for a = 1:length(scan_angles)
    for m = 1:num_mics
        path = mic_pos(m,1)*cosd(scan_angles(a)) + mic_pos(m,2)*sind(scan_angles(a));
        V_coarse(m, a) = exp(-1j * k_1k * path);
    end
end

P_coarse = zeros(1, length(scan_angles));
for a = 1:length(scan_angles)
    v = V_coarse(:, a);
    P_coarse(a) = real(1 / (v' * R_1k_inv * v));
end

Rs_1k = zeros(num_mics);
Rn_1k = zeros(num_mics);
for a = 1:length(scan_angles)
    v = V_coarse(:, a);
    outer = P_coarse(a) * (v * v');
    if in_fov_coarse(a)
        Rs_1k = Rs_1k + outer;
    else
        Rn_1k = Rn_1k + outer;
    end
end
Rs_1k = Rs_1k + 1e-6*eye(num_mics);
Rn_1k = Rn_1k + 1e-6*eye(num_mics);

[V_gev, D_gev] = eig(Rs_1k, Rn_1k);
[~, max_i] = max(real(diag(D_gev)));
w_gev = V_gev(:, max_i);
w_gev = w_gev / norm(w_gev);

% Beam pattern
beam = zeros(1, length(plot_angles));
for a = 1:length(plot_angles)
    beam(a) = abs(w_gev' * V_plot(:, a))^2;
end
beam_dB = 10*log10(beam/max(beam) + 1e-10);

%% Visualization
figure('Position', [50, 50, 1400, 700], 'Color', 'w');

% Array geometry
subplot(2, 3, 1);
plot(mic_pos(:,1)*100, mic_pos(:,2)*100, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.3 0.6 0.9], 'LineWidth', 2);
hold on;
th = linspace(0, 2*pi, 100);
plot(radius*100*cos(th), radius*100*sin(th), 'b--');
arr_len = radius*150;
quiver(0, 0, arr_len*cosd(target_angle), arr_len*sind(target_angle), 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, arr_len*cosd(interf_angle), arr_len*sind(interf_angle), 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
xlabel('X (cm)'); ylabel('Y (cm)');
title('Array Geometry');
legend('Mics', 'Array', 'Target', 'Interference', 'Location', 'best');
axis equal; grid on;
xlim([-10 10]); ylim([-10 10]);

% Input spectrogram
subplot(2, 3, 2);
spectrogram(ref_input, hann(256), 192, 256, fs, 'yaxis');
title('Input (Mic 1) - Noisy');
ylim([0 4]);
colorbar;

% Output spectrogram
subplot(2, 3, 3);
spectrogram(output, hann(256), 192, 256, fs, 'yaxis');
title('Output - GEV Beamformed');
ylim([0 4]);
colorbar;

% Time domain
subplot(2, 3, 4);
plot(t, ref_input, 'b', 'LineWidth', 0.5);
hold on;
plot(t, output, 'r', 'LineWidth', 0.5);
xlabel('Time (s)'); ylabel('Amplitude');
title('Waveform Comparison');
legend('Input', 'Output');
grid on; xlim([0 duration]);

% Beam pattern (Cartesian)
subplot(2, 3, 5);
plot(plot_angles, beam_dB, 'b-', 'LineWidth', 2);
hold on;
xline(target_angle, 'g--', 'LineWidth', 2);
xline(interf_angle, 'r--', 'LineWidth', 2);
fill([FOV(1), FOV(2), FOV(2), FOV(1)], [-50 -50 10 10], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
xlabel('Angle (°)'); ylabel('Gain (dB)');
title('Beam Pattern at 1 kHz');
legend('Pattern', 'Target', 'Interference', 'FOV');
xlim([-180 180]); ylim([-40 5]);
grid on;

% Beam pattern (Polar)
subplot(2, 3, 6);
beam_lin = 10.^(beam_dB/20);
beam_lin = max(beam_lin, 0.01);
polarplot(deg2rad(plot_angles), beam_lin, 'b-', 'LineWidth', 2);
hold on;
polarplot(deg2rad(target_angle), beam_lin(plot_angles == target_angle), 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
polarplot(deg2rad(interf_angle), beam_lin(plot_angles == interf_angle), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
title('Polar Beam Pattern');
rlim([0 1.2]);

sgtitle('GEV Beamformer: Audiovisual Zooming Demo', 'FontSize', 14, 'FontWeight', 'bold');

%% Performance Metrics
[~, t_idx] = min(abs(plot_angles - target_angle));
[~, i_idx] = min(abs(plot_angles - interf_angle));
suppression = beam_dB(t_idx) - beam_dB(i_idx);

fprintf('\n=== Results ===\n');
fprintf('Gain at target (%d°): %.1f dB\n', target_angle, beam_dB(t_idx));
fprintf('Gain at interference (%d°): %.1f dB\n', interf_angle, beam_dB(i_idx));
fprintf('Suppression: %.1f dB\n', suppression);

%% Audio Playback
fprintf('\nPlaying input...\n');
soundsc(ref_input, fs);
pause(duration + 0.5);

fprintf('Playing output (zoomed)...\n');
soundsc(output, fs);
pause(duration + 0.5);

fprintf('\nDemo complete!\n');

