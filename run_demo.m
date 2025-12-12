%% DEMO - Audiovisual Zooming Quick Test
%  Run this to quickly verify the GEV beamformer works
%  Target at 0°, Interference at 90°, FOV ±15°

clear; close all; clc;
fprintf('=== Quick Demo: GEV Beamformer ===\n');

%% Setup
fs = 16000; c = 343; dur = 2; N = dur*fs; t = (0:N-1)'/fs;
num_mics = 6; r = 0.05;
mic_pos = r * [cosd((0:5)*60)', sind((0:5)*60)'];

%% Create Test Signals (Natural sounding - easier on ears)
% Target: Speech-like harmonics at 0°
f0 = 200;
target = zeros(N, 1);
for h = 1:6
    target = target + sin(2*pi*f0*h*t + rand*2*pi) / h;
end
target = target .* (0.5 + 0.4*sin(2*pi*3*t));  % AM modulation
target = 0.5 * target / max(abs(target));      % 50% volume

% Interference: Soft filtered noise at 90°
interf = bandpass(randn(N, 1), [400 2000], fs);
interf = 0.4 * interf / max(abs(interf));      % 40% volume

%% Apply Delays
mic_sig = zeros(N, num_mics);
freqs = (0:N-1)'*fs/N; freqs(freqs>fs/2) = freqs(freqs>fs/2)-fs;
for m = 1:num_mics
    tau_t = dot(mic_pos(m,:), [1 0])/c;  % 0°
    tau_i = dot(mic_pos(m,:), [0 1])/c;  % 90°
    mic_sig(:,m) = real(ifft(fft(target).*exp(-1j*2*pi*freqs*tau_t))) + ...
                   real(ifft(fft(interf).*exp(-1j*2*pi*freqs*tau_i))) + 0.01*randn(N,1);
end

%% STFT
[S, f, ~] = stft(mic_sig, fs, 'Window', hann(512,'periodic'), 'OverlapLength', 384, 'FFTLength', 512);
[~, nt, ~] = size(S);
bins = find(f >= 300 & f <= 3400);
Out = mean(S, 3);

%% GEV Beamforming
fov = [-15 15]; angles = -180:20:160; reg = 1e-6;

for i = 1:length(bins)
    b = bins(i); k = 2*pi*f(b)/c;
    X = squeeze(S(b,:,:)).';
    R = (X*X')/nt + reg*eye(num_mics);
    
    V = exp(-1j*k*(mic_pos(:,1)*cosd(angles) + mic_pos(:,2)*sind(angles)));
    Ri = inv(R);
    P = max(real(1./sum(conj(V).*(Ri*V),1)), 0);
    
    in_fov = angles >= fov(1) & angles <= fov(2);
    Rs = V(:,in_fov)*diag(P(in_fov))*V(:,in_fov)' + reg*eye(num_mics);
    Rn = V(:,~in_fov)*diag(P(~in_fov))*V(:,~in_fov)' + reg*eye(num_mics);
    
    [Vecs, D] = eig(Rs, Rn);
    [~, mi] = max(real(diag(D)));
    w = Vecs(:,mi)/norm(Vecs(:,mi));
    Out(b,:) = w' * X;
end

%% Reconstruct
y = real(istft(Out, fs, 'Window', hann(512,'periodic'), 'OverlapLength', 384, 'FFTLength', 512));
y = y(1:min(N,length(y))); y = y/max(abs(y));
ref = mic_sig(:,1)/max(abs(mic_sig(:,1)));

%% Beam Pattern
k1k = 2*pi*1000/c; pa = -180:1:179;
V1k = exp(-1j*k1k*(mic_pos(:,1)*cosd(pa) + mic_pos(:,2)*sind(pa)));
Vs = exp(-1j*k1k*(mic_pos(:,1)*cosd(angles) + mic_pos(:,2)*sind(angles)));
[~,b1k] = min(abs(f-1000)); X1k = squeeze(S(b1k,:,:)).';
R1k = (X1k*X1k')/nt + reg*eye(num_mics); Ri1k = inv(R1k);
Ps = max(real(1./sum(conj(Vs).*(Ri1k*Vs),1)),0);
in_fov = angles>=fov(1) & angles<=fov(2);
Rs1k = Vs(:,in_fov)*diag(Ps(in_fov))*Vs(:,in_fov)' + reg*eye(num_mics);
Rn1k = Vs(:,~in_fov)*diag(Ps(~in_fov))*Vs(:,~in_fov)' + reg*eye(num_mics);
[Vg,Dg] = eig(Rs1k,Rn1k); [~,mi] = max(real(diag(Dg))); wg = Vg(:,mi)/norm(Vg(:,mi));
bp_dB = 10*log10(abs(wg'*V1k).^2/max(abs(wg'*V1k).^2)+1e-10);

%% Results
suppression = bp_dB(pa==0) - bp_dB(pa==90);
fprintf('\nTarget gain (0°):  %.1f dB\n', bp_dB(pa==0));
fprintf('Interf gain (90°): %.1f dB\n', bp_dB(pa==90));
fprintf('SUPPRESSION:       %.1f dB\n', suppression);

%% Plot
figure('Position',[100 100 900 400],'Color','w');

subplot(1,2,1);
spectrogram(ref,hann(256),192,256,fs,'yaxis'); title('INPUT (Speech + Noise mixed)'); ylim([0 4]);

subplot(1,2,2);
spectrogram(y,hann(256),192,256,fs,'yaxis'); title('OUTPUT (Noise suppressed)'); ylim([0 4]);

sgtitle(sprintf('Demo: %.1f dB suppression at 90°', suppression), 'FontWeight', 'bold');

%% Play Audio (reduced volume)
fprintf('\nPlaying INPUT (speech + noise)...\n');
soundsc(ref * 0.5, fs); pause(dur+0.3);
fprintf('Playing OUTPUT (noise suppressed)...\n');
soundsc(y * 0.5, fs);

fprintf('\n=== Demo Complete ===\n');

