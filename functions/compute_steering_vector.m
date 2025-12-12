function v = compute_steering_vector(mic_positions, angle_deg, wavelength)
% COMPUTE_STEERING_VECTOR Compute array steering vector for plane wave
%
%   v = compute_steering_vector(mic_positions, angle_deg, wavelength)
%
%   Computes the steering vector for a plane wave arriving from a specified
%   direction at a given frequency (wavelength).
%
%   Inputs:
%       mic_positions - [num_mics x 2] matrix of microphone positions (x, y) in meters
%       angle_deg     - Direction of arrival in degrees (0 = +x axis, counter-clockwise)
%       wavelength    - Acoustic wavelength in meters (c/f)
%
%   Output:
%       v - [num_mics x 1] complex steering vector
%
%   The steering vector represents the phase relationship between microphones
%   for a plane wave from the specified direction:
%       v(m) = exp(-j * k * d_m . u_theta)
%
%   where k = 2*pi/wavelength, d_m is mic position, u_theta is unit direction vector.
%
%   Example:
%       % 4-mic linear array, 1kHz wave from 30 degrees
%       mic_pos = [0, 0; 0.05, 0; 0.10, 0; 0.15, 0];
%       v = compute_steering_vector(mic_pos, 30, 343/1000);
%
%   See also: compute_covariance_matrix, compute_mpdr_power

    % Validate inputs
    if size(mic_positions, 2) ~= 2
        error('mic_positions must be [num_mics x 2] matrix');
    end
    
    if wavelength <= 0
        error('wavelength must be positive');
    end
    
    num_mics = size(mic_positions, 1);
    
    % Wave number
    k = 2 * pi / wavelength;
    
    % Unit vector pointing toward source direction
    ux = cosd(angle_deg);
    uy = sind(angle_deg);
    
    % Compute phase delays for each microphone
    v = zeros(num_mics, 1);
    
    for m = 1:num_mics
        % Path difference from array center (origin assumed)
        % Positive path_diff means wave arrives later at this mic
        path_diff = mic_positions(m, 1) * ux + mic_positions(m, 2) * uy;
        
        % Phase shift (negative because of incoming wave convention)
        v(m) = exp(-1j * k * path_diff);
    end
end

