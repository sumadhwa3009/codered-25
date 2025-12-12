function [stft_matrix, freq_axis, time_axis] = compute_stft(signal, fs, fft_size, hop_size, window)
% COMPUTE_STFT Compute Short-Time Fourier Transform
%
%   [stft_matrix, freq_axis, time_axis] = compute_stft(signal, fs, fft_size, hop_size)
%   [stft_matrix, freq_axis, time_axis] = compute_stft(signal, fs, fft_size, hop_size, window)
%
%   Computes the STFT of a multi-channel signal using overlap-add framework.
%
%   Inputs:
%       signal   - [num_samples x num_channels] input signal matrix
%       fs       - Sampling frequency (Hz)
%       fft_size - FFT size (window length)
%       hop_size - Hop size between frames (samples)
%       window   - (optional) Analysis window. Default: Hann window
%
%   Outputs:
%       stft_matrix - [num_freq_bins x num_frames x num_channels] complex STFT
%       freq_axis   - [num_freq_bins x 1] frequency values (Hz)
%       time_axis   - [num_frames x 1] time values (seconds)
%
%   Notes:
%       - Only positive frequencies are returned (DC to Nyquist)
%       - num_freq_bins = fft_size/2 + 1
%
%   Example:
%       [S, f, t] = compute_stft(audio, 16000, 512, 128);
%       imagesc(t, f, 20*log10(abs(S(:,:,1))));
%
%   See also: compute_istft, spectrogram

    % Get dimensions
    [num_samples, num_channels] = size(signal);
    
    % Default window
    if nargin < 5
        window = hann(fft_size, 'periodic');
    end
    
    % Ensure window is column vector
    window = window(:);
    
    if length(window) ~= fft_size
        error('Window length must equal fft_size');
    end
    
    % Calculate number of frames
    num_frames = floor((num_samples - fft_size) / hop_size) + 1;
    
    % Number of frequency bins (positive frequencies only)
    num_freq_bins = fft_size/2 + 1;
    
    % Initialize output
    stft_matrix = zeros(num_freq_bins, num_frames, num_channels);
    
    % Compute STFT for each channel
    for ch = 1:num_channels
        for frame = 1:num_frames
            % Extract frame
            start_idx = (frame - 1) * hop_size + 1;
            end_idx = start_idx + fft_size - 1;
            
            if end_idx <= num_samples
                % Apply window
                segment = signal(start_idx:end_idx, ch) .* window;
                
                % Compute FFT
                spectrum = fft(segment, fft_size);
                
                % Store positive frequencies
                stft_matrix(:, frame, ch) = spectrum(1:num_freq_bins);
            end
        end
    end
    
    % Compute frequency axis
    freq_axis = (0:num_freq_bins-1)' * fs / fft_size;
    
    % Compute time axis
    time_axis = ((0:num_frames-1)' * hop_size + fft_size/2) / fs;
end

