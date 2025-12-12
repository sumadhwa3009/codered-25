%% DEMO - Audiovisual Zooming (7-Mic Array)

clear; close all; clc;
fprintf('=== GEV Beamformer Demo (7-Mic Array) ===\n');

%% Setup
fs = 16000;
c = 343; dur = 2; N = dur*fs; t = (0:N-1)'/fs;
num_mics = 7;
r = 0.05;
mic_angles = [0, 60, 120, 180, 240, 300];  % degrees for circular mics
mic_pos = zeros(num_mics, 2);
for i = 1:6
    mic_pos(i, :) = r * [cosd(mic_angles(i)), sind(mic_angles(i))];
end
mic_pos(7, :) = [0, 0];  

fprintf('Array: 6 circular mics + 1 center mic\n');
fprintf('Radius: %.0f cm\n', r*100);

%% Create Signals
target = sin(2*pi*300*t) + 0.7*sin(2*pi*500*t) + 0.5*sin(2*pi*700*t);
target = target .* (0.6 + 0.4*sin(2*pi*2*t));
target = target / max(abs(target));
interf = sin(2*pi*1200*t) + 0.8*sin(2*pi*1500*t) + 0.6*sin(2*pi*1800*t);
interf = interf / max(abs(interf));

%% Apply Delays to all 7 mics
mic_sig = zeros(N, num_mics);
freqs = (0:N-1)'*fs/N; freqs(freqs>fs/2) = freqs(freqs>fs/2)-fs;

for m = 1:num_mics
    tau_t = (mic_pos(m,1)*cosd(0) + mic_pos(m,2)*sind(0))/c;    
    tau_i = (mic_pos(m,1)*cosd(90) + mic_pos(m,2)*sind(90))/c;  
    mic_sig(:,m) = real(ifft(fft(target).*exp(-1j*2*pi*freqs*tau_t))) + ...
                   real(ifft(fft(interf).*exp(-1j*2*pi*freqs*tau_i))) + 0.01*randn(N,1);
end
%% STFT
[S, f, ~] = stft(mic_sig, fs, 'Window', hann(256,'periodic'), 'OverlapLength', 192, 'FFTLength', 256);
[~, nt, ~] = size(S);
bins = find(f >= 200 & f <= 3000);
Out = mean(S, 3);

%% GEV Beamforming
fov = [-20 20]; angles = -180:30:150; reg = 1e-5;

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
y = real(istft(Out, fs, 'Window', hann(256,'periodic'), 'OverlapLength', 192, 'FFTLength', 256));
y = y(1:min(N,length(y)));
y = 2 * y / max(abs(y));
y = min(max(y, -1), 1);
ref = mic_sig(:,1) / max(abs(mic_sig(:,1)));

%% Beam pattern
k1k = 2*pi*800/c; pa = -180:2:178;
V1k = exp(-1j*k1k*(mic_pos(:,1)*cosd(pa) + mic_pos(:,2)*sind(pa)));
[~,b1k] = min(abs(f-800)); X1k = squeeze(S(b1k,:,:)).';
R1k = (X1k*X1k')/nt + reg*eye(num_mics); Ri1k = inv(R1k);
Vs = exp(-1j*k1k*(mic_pos(:,1)*cosd(angles) + mic_pos(:,2)*sind(angles)));
Ps = max(real(1./sum(conj(Vs).*(Ri1k*Vs),1)),0);
in_fov = angles>=fov(1) & angles<=fov(2);
Rs1k = Vs(:,in_fov)*diag(Ps(in_fov))*Vs(:,in_fov)' + reg*eye(num_mics);
Rn1k = Vs(:,~in_fov)*diag(Ps(~in_fov))*Vs(:,~in_fov)' + reg*eye(num_mics);
[Vg,Dg] = eig(Rs1k,Rn1k); [~,mi] = max(real(diag(Dg))); wg = Vg(:,mi)/norm(Vg(:,mi));
bp = abs(wg'*V1k).^2; bp_dB = 10*log10(bp/max(bp)+1e-10);
suppression = bp_dB(pa==0) - bp_dB(pa==90);

fprintf('Suppression: %.1f dB\n', suppression);

%% Plot
figure('Position',[100 100 1000 400],'Color','w');
plot(mic_pos(7,1)*100, mic_pos(7,2)*100, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
th = linspace(0,2*pi,100); plot(r*100*cos(th), r*100*sin(th), 'b--');
axis equal; grid on; xlim([-8 8]); ylim([-8 8]);
xlabel('X (cm)'); ylabel('Y (cm)');
title('7-Mic Array'); legend('Circular', 'Center', 'Location', 'best');

subplot(1,3,1);
spectrogram(ref,128,96,128,fs,'yaxis'); title('INPUT'); ylim([0 3]);

subplot(1,3,2);
spectrogram(y,128,96,128,fs,'yaxis'); title('OUTPUT'); ylim([0 3]);

sgtitle(sprintf('7-Mic GEV Demo: %.0f dB suppression', suppression), 'FontWeight','bold');

%% Play Audio
fprintf('\n>>> Playing INPUT...\n');
sound(ref, fs); pause(dur + 0.5);
fprintf('>>> Playing OUTPUT...\n');
sound(y, fs); pause(dur + 0.5);

fprintf('\nDone!\n');