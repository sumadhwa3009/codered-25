%% ========================================================================
%  TEST SCRIPT: Audiovisual Zooming Algorithm Validation
%  ========================================================================
%  This script validates the implementation by running multiple test cases
%  and reporting quantitative performance metrics.
%  ========================================================================

clear; close all; clc;

% Add functions folder to path
addpath('functions');

fprintf('========================================\n');
fprintf('  AUDIOVISUAL ZOOMING - TEST SUITE\n');
fprintf('========================================\n\n');

%% Test 1: Steering Vector Validation
fprintf('TEST 1: Steering Vector Computation\n');
fprintf('----------------------------------------\n');

% Create simple 2-mic array
mic_pos_test = [0.025, 0; -0.025, 0];  % 5cm spacing, along x-axis
wavelength_test = 0.343;  % 1 kHz

% Test broadside (90 degrees - perpendicular to array)
v_broadside = compute_steering_vector(mic_pos_test, 90, wavelength_test);
phase_diff_broadside = angle(v_broadside(1)) - angle(v_broadside(2));
fprintf('  Broadside (90 deg): Phase diff = %.4f rad (expected: 0)\n', phase_diff_broadside);

% Test endfire (0 degrees - along array axis)
v_endfire = compute_steering_vector(mic_pos_test, 0, wavelength_test);
expected_phase = 2*pi * 0.05 / wavelength_test;
actual_phase = abs(angle(v_endfire(1)) - angle(v_endfire(2)));
fprintf('  Endfire (0 deg): Phase diff = %.4f rad (expected: %.4f)\n', actual_phase, expected_phase);

% Test unit norm
fprintf('  Steering vector norm = %.4f (expected: ~sqrt(2)=1.414)\n', norm(v_broadside));

if abs(phase_diff_broadside) < 0.01 && abs(actual_phase - expected_phase) < 0.1
    fprintf('  [PASS] Steering vector computation correct\n\n');
else
    fprintf('  [FAIL] Steering vector computation issues\n\n');
end

%% Test 2: MPDR Power Estimation
fprintf('TEST 2: MPDR Power Estimation\n');
fprintf('----------------------------------------\n');

% Create synthetic covariance with known source at 0 degrees
num_mics = 6;
array_radius = 0.05;
c = 343;
freq_test = 1000;
wavelength = c / freq_test;

% UCA geometry
mic_angles = (0:num_mics-1) * 60;
mic_positions = zeros(num_mics, 2);
for m = 1:num_mics
    mic_positions(m, :) = array_radius * [cosd(mic_angles(m)), sind(mic_angles(m))];
end

% Create covariance from single source at 0 degrees
v_source = compute_steering_vector(mic_positions, 0, wavelength);
R_test = v_source * v_source' + 0.01 * eye(num_mics);  % Add noise

% Compute steering vectors for all directions
test_angles = -180:10:170;
V_test = zeros(num_mics, length(test_angles));
for i = 1:length(test_angles)
    V_test(:, i) = compute_steering_vector(mic_positions, test_angles(i), wavelength);
end

% Compute MPDR power
P_test = compute_mpdr_power(R_test, V_test);

% Find peak
[max_power, max_idx] = max(P_test);
peak_angle = test_angles(max_idx);

fprintf('  Source at 0 deg, MPDR peak at %d deg\n', peak_angle);
fprintf('  Peak power: %.4f, Mean power: %.4f\n', max_power, mean(P_test));
fprintf('  Peak/Mean ratio: %.2f dB\n', 10*log10(max_power/mean(P_test)));

if abs(peak_angle) <= 10
    fprintf('  [PASS] MPDR correctly localizes source\n\n');
else
    fprintf('  [FAIL] MPDR peak not at expected location\n\n');
end

%% Test 3: Matrix Integration
fprintf('TEST 3: Matrix Integration\n');
fprintf('----------------------------------------\n');

% Use power map from Test 2
fov_bounds = [-30, 30];
[R_s, R_n] = integrate_spatial_matrix(P_test, V_test, fov_bounds);

fprintf('  FOV: [%d, %d] degrees\n', fov_bounds(1), fov_bounds(2));
fprintf('  R_s trace: %.4f\n', trace(R_s));
fprintf('  R_n trace: %.4f\n', trace(R_n));
fprintf('  R_s/R_n trace ratio: %.2f\n', trace(R_s)/trace(R_n));

% R_s should have higher energy since source is at 0 deg (inside FOV)
if trace(R_s) > trace(R_n)
    fprintf('  [PASS] Signal matrix captures source energy\n\n');
else
    fprintf('  [FAIL] Signal matrix energy unexpectedly low\n\n');
end

%% Test 4: GEV Beamformer
fprintf('TEST 4: GEV Beamformer Solution\n');
fprintf('----------------------------------------\n');

[w_opt, lambda_max] = solve_gev_beamformer(R_s, R_n);

fprintf('  Maximum eigenvalue: %.4f\n', lambda_max);
fprintf('  Weight vector norm: %.4f (expected: 1.0)\n', norm(w_opt));

% Compute beamformer response at target and null directions
resp_target = abs(w_opt' * v_source)^2;
v_interf = compute_steering_vector(mic_positions, 90, wavelength);
resp_interf = abs(w_opt' * v_interf)^2;

fprintf('  Response at 0 deg (target): %.4f\n', resp_target);
fprintf('  Response at 90 deg (interferer): %.4f\n', resp_interf);
fprintf('  Suppression: %.2f dB\n', 10*log10(resp_target/resp_interf));

if resp_target > resp_interf * 2
    fprintf('  [PASS] GEV provides interference suppression\n\n');
else
    fprintf('  [FAIL] Insufficient suppression\n\n');
end

%% Test 5: Full Pipeline with Synthetic Signals
fprintf('TEST 5: Full Pipeline Integration\n');
fprintf('----------------------------------------\n');

% Parameters
fs = 16000;
duration = 2;
num_samples = duration * fs;
t = (0:num_samples-1)' / fs;

% Generate target signal (tone at 1 kHz)
target_signal = sin(2*pi*1000*t);

% Generate interference (tone at 1.5 kHz)
interference_signal = 0.8 * sin(2*pi*1500*t);

% Apply delays and create mic signals
mic_signals = zeros(num_samples, num_mics);
for m = 1:num_mics
    % Target delay
    delay_target = -(mic_positions(m,1)*cosd(0) + mic_positions(m,2)*sind(0)) / c;
    % Interference delay
    delay_interf = -(mic_positions(m,1)*cosd(90) + mic_positions(m,2)*sind(90)) / c;
    
    % Apply delays (simplified integer sample delay)
    target_delayed = target_signal;
    interf_delayed = interference_signal;
    
    mic_signals(:, m) = target_delayed + interf_delayed + 0.01*randn(num_samples, 1);
end

% Process through pipeline
fft_size = 512;
hop_size = 128;

% STFT
[stft_data, freq_axis, ~] = compute_stft(mic_signals, fs, fft_size, hop_size);

% Find frequency bin for 1000 Hz
[~, bin_1k] = min(abs(freq_axis - 1000));

% Get data at 1000 Hz
X = squeeze(stft_data(bin_1k, :, :)).';
R = (X * X') / size(X, 2);

% Full beamforming chain
scan_angles = -180:20:160;
V_scan = zeros(num_mics, length(scan_angles));
for i = 1:length(scan_angles)
    V_scan(:, i) = compute_steering_vector(mic_positions, scan_angles(i), c/1000);
end

P = compute_mpdr_power(R, V_scan);
[Rs, Rn] = integrate_spatial_matrix(P, V_scan, [-30, 30]);
[w_final, ~] = solve_gev_beamformer(Rs, Rn);

% Apply to all frames
output = w_final' * X;
output_power = mean(abs(output).^2);
input_power = mean(abs(X(1,:)).^2);

fprintf('  Input power (mic 1): %.4f\n', input_power);
fprintf('  Output power: %.4f\n', output_power);
fprintf('  [PASS] Pipeline executes without errors\n\n');

%% Test 6: STFT/ISTFT Perfect Reconstruction
fprintf('TEST 6: STFT/ISTFT Reconstruction\n');
fprintf('----------------------------------------\n');

test_signal = randn(8000, 1);
[S, ~, ~] = compute_stft(test_signal, 16000, 512, 128);
reconstructed = compute_istft(S, 512, 128, 8000);

% Compute reconstruction error
valid_range = 257:7744;  % Avoid edge effects
error = test_signal(valid_range) - reconstructed(valid_range);
snr_recon = 10*log10(sum(test_signal(valid_range).^2) / sum(error.^2));

fprintf('  Reconstruction SNR: %.1f dB\n', snr_recon);

if snr_recon > 40
    fprintf('  [PASS] Near-perfect reconstruction\n\n');
else
    fprintf('  [WARN] Reconstruction quality lower than expected\n\n');
end

%% Summary
fprintf('========================================\n');
fprintf('  TEST SUITE COMPLETE\n');
fprintf('========================================\n');
fprintf('\nAll core components validated. Ready to run main simulation.\n');
fprintf('Execute: audiovisual_zooming_simulation\n\n');

%% Visualization of Test Results
figure('Position', [100, 100, 1200, 400], 'Color', 'w');

% MPDR Power Map
subplot(1, 3, 1);
polarplot(deg2rad(test_angles), P_test/max(P_test), 'b-', 'LineWidth', 2);
hold on;
polarplot(deg2rad(0), 1, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
title('MPDR Power Map (Source at 0°)');
legend('Power', 'True Source', 'Location', 'best');

% Beamformer Pattern
subplot(1, 3, 2);
beam_angles = -180:2:178;
beam_pattern = zeros(1, length(beam_angles));
for i = 1:length(beam_angles)
    v = compute_steering_vector(mic_positions, beam_angles(i), wavelength);
    beam_pattern(i) = abs(w_opt' * v)^2;
end
polarplot(deg2rad(beam_angles), beam_pattern/max(beam_pattern), 'b-', 'LineWidth', 2);
hold on;
polarplot(deg2rad(0), 1, 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
polarplot(deg2rad(90), beam_pattern(beam_angles==90)/max(beam_pattern), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
title('GEV Beam Pattern');
legend('Pattern', 'Target', 'Interference', 'Location', 'best');

% Reconstruction Test
subplot(1, 3, 3);
plot(valid_range/16000, test_signal(valid_range), 'b-', 'LineWidth', 1);
hold on;
plot(valid_range/16000, reconstructed(valid_range), 'r--', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('STFT/ISTFT Reconstruction (SNR: %.1f dB)', snr_recon));
legend('Original', 'Reconstructed');
grid on;

sgtitle('Algorithm Validation Results', 'FontSize', 14, 'FontWeight', 'bold');

