function [R_s, R_n] = integrate_spatial_matrix(power_map, steering_vectors, fov_bounds)
% INTEGRATE_SPATIAL_MATRIX Compute signal and noise matrices via angular integration
%
%   [R_s, R_n] = integrate_spatial_matrix(power_map, steering_vectors, fov_bounds)
%
%   Implements Eq. (11) from the Audiovisual Zooming paper, integrating the
%   weighted outer products of steering vectors over signal (FOV) and noise
%   (outside FOV) regions:
%
%       R_s = sum_{theta in FOV}     P(theta) * v_theta * v_theta^H
%       R_n = sum_{theta not in FOV} P(theta) * v_theta * v_theta^H
%
%   Inputs:
%       power_map        - [1 x num_angles] MPDR power estimates
%       steering_vectors - [num_mics x num_angles] steering vector matrix
%       fov_bounds       - [1 x 2] Field of View boundaries [min_angle, max_angle] in degrees
%
%   Outputs:
%       R_s - [num_mics x num_mics] Signal covariance matrix (inside FOV)
%       R_n - [num_mics x num_mics] Noise covariance matrix (outside FOV)
%
%   Notes:
%       - Angles are assumed to be evenly spaced from -180 to +180 degrees
%       - The FOV is defined as the angular region between fov_bounds(1) and fov_bounds(2)
%       - FOV wrapping around 180/-180 is supported
%
%   Example:
%       fov = [-30, 30];  % +/- 30 degrees
%       [R_s, R_n] = integrate_spatial_matrix(P, V, fov);
%
%   See also: compute_mpdr_power, solve_gev_beamformer

    % Extract dimensions
    num_mics = size(steering_vectors, 1);
    num_angles = size(steering_vectors, 2);
    
    % Validate inputs
    if length(power_map) ~= num_angles
        error('power_map length must match number of steering vectors');
    end
    
    if length(fov_bounds) ~= 2
        error('fov_bounds must be [min_angle, max_angle]');
    end
    
    % Compute angle values (assuming uniform sampling from -180 to 180)
    angles = linspace(-180, 180 - 360/num_angles, num_angles);
    
    % Determine which angles are inside FOV
    fov_min = fov_bounds(1);
    fov_max = fov_bounds(2);
    
    if fov_min <= fov_max
        % Normal case: FOV doesn't wrap around
        in_fov = (angles >= fov_min) & (angles <= fov_max);
    else
        % Wrapped case: FOV crosses the -180/+180 boundary
        in_fov = (angles >= fov_min) | (angles <= fov_max);
    end
    
    % Initialize output matrices
    R_s = zeros(num_mics, num_mics);
    R_n = zeros(num_mics, num_mics);
    
    % Integrate over all angles
    for a = 1:num_angles
        v = steering_vectors(:, a);
        P = power_map(a);
        
        % Weighted outer product
        weighted_matrix = P * (v * v');
        
        if in_fov(a)
            % Add to signal matrix
            R_s = R_s + weighted_matrix;
        else
            % Add to noise matrix
            R_n = R_n + weighted_matrix;
        end
    end
    
    % Ensure Hermitian symmetry (numerical cleanup)
    R_s = (R_s + R_s') / 2;
    R_n = (R_n + R_n') / 2;
end

