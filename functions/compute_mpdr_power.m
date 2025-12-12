function P = compute_mpdr_power(R, steering_vectors, reg_param)
% COMPUTE_MPDR_POWER Compute MPDR spatial power spectrum
%
%   P = compute_mpdr_power(R, steering_vectors)
%   P = compute_mpdr_power(R, steering_vectors, reg_param)
%
%   Computes the Minimum Power Distortionless Response (MPDR) power estimate
%   for multiple directions using Eq. (10) from the Audiovisual Zooming paper:
%
%       P(theta) = [v_theta^H * R^(-1) * v_theta]^(-1)
%
%   Inputs:
%       R                - [num_mics x num_mics] spatial covariance matrix
%       steering_vectors - [num_mics x num_angles] matrix of steering vectors
%       reg_param        - (optional) regularization parameter for matrix inversion
%                          Default: 1e-6
%
%   Output:
%       P - [1 x num_angles] MPDR power estimates for each direction
%
%   The MPDR estimator (also known as Capon beamformer) provides a spatial
%   power spectrum that is more focused than conventional beamforming,
%   with better resolution and interference rejection.
%
%   Example:
%       % Compute power map for 6-mic array
%       R = X * X' / num_frames;  % Sample covariance
%       angles = -180:10:170;
%       for i = 1:length(angles)
%           V(:,i) = compute_steering_vector(mic_pos, angles(i), wavelength);
%       end
%       P = compute_mpdr_power(R, V);
%
%   See also: compute_steering_vector, solve_gev_beamformer

    % Default regularization
    if nargin < 3
        reg_param = 1e-6;
    end
    
    % Validate inputs
    num_mics = size(R, 1);
    if size(R, 2) ~= num_mics
        error('R must be a square matrix');
    end
    
    if size(steering_vectors, 1) ~= num_mics
        error('steering_vectors must have num_mics rows');
    end
    
    num_angles = size(steering_vectors, 2);
    
    % Add regularization for numerical stability
    R_reg = R + reg_param * eye(num_mics);
    
    % Compute matrix inverse
    try
        R_inv = inv(R_reg);
    catch
        warning('Matrix inversion failed, using pseudo-inverse');
        R_inv = pinv(R_reg);
    end
    
    % Compute MPDR power for each direction
    P = zeros(1, num_angles);
    
    for a = 1:num_angles
        v = steering_vectors(:, a);
        
        % MPDR formula: P(theta) = 1 / (v^H * R^-1 * v)
        denom = v' * R_inv * v;
        
        % Take real part (should be real for proper covariance matrix)
        P(a) = real(1 / denom);
    end
    
    % Ensure non-negative power (numerical protection)
    P = max(P, 0);
end

