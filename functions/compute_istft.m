function signal = compute_istft(stft_matrix, fft_size, hop_size, num_samples, window)
% COMPUTE_ISTFT Compute Inverse Short-Time Fourier Transform
%
%   signal = compute_istft(stft_matrix, fft_size, hop_size, num_samples)
%   signal = compute_istft(stft_matrix, fft_size, hop_size, num_samples, window)
%
%   Reconstructs a time-domain signal from its STFT representation using
%   overlap-add synthesis.
%
%   Inputs:
%       stft_matrix - [num_freq_bins x num_frames] complex STFT (single channel)
%       fft_size    - FFT size used in forward STFT
%       hop_size    - Hop size between frames (samples)
%       num_samples - Desired output signal length (samples)
%       window      - (optional) Synthesis window. Default: Hann window
%
%   Output:
%       signal - [num_samples x 1] reconstructed time-domain signal
%
%   Notes:
%       - Input should contain only positive frequencies (DC to Nyquist)
%       - Overlap-add with proper window normalization ensures perfect reconstruction
%
%   Example:
%       y = compute_istft(S, 512, 128, length(x));
%
%   See also: compute_stft

    % Get dimensions
    [num_freq_bins, num_frames] = size(stft_matrix);
    
    % Validate frequency bins
    expected_bins = fft_size/2 + 1;
    if num_freq_bins ~= expected_bins
        error('STFT should have %d frequency bins for fft_size=%d', expected_bins, fft_size);
    end
    
    % Default synthesis window
    if nargin < 5
        window = hann(fft_size, 'periodic');
    end
    
    % Ensure window is column vector
    window = window(:);
    
    % Initialize output buffers
    signal = zeros(num_samples, 1);
    window_sum = zeros(num_samples, 1);
    
    % Process each frame
    for frame = 1:num_frames
        start_idx = (frame - 1) * hop_size + 1;
        end_idx = start_idx + fft_size - 1;
        
        if end_idx <= num_samples
            % Get STFT frame
            spectrum_pos = stft_matrix(:, frame);
            
            % Reconstruct full spectrum (conjugate symmetric for real signal)
            full_spectrum = zeros(fft_size, 1);
            full_spectrum(1:num_freq_bins) = spectrum_pos;
            
            % Negative frequencies (conjugate symmetric)
            full_spectrum(num_freq_bins+1:end) = conj(flipud(spectrum_pos(2:end-1)));
            
            % IFFT
            frame_signal = real(ifft(full_spectrum));
            
            % Apply synthesis window and overlap-add
            signal(start_idx:end_idx) = signal(start_idx:end_idx) + frame_signal .* window;
            window_sum(start_idx:end_idx) = window_sum(start_idx:end_idx) + window.^2;
        end
    end
    
    % Normalize by accumulated window
    % Avoid division by zero
    valid_idx = window_sum > 1e-10;
    signal(valid_idx) = signal(valid_idx) ./ window_sum(valid_idx);
end

