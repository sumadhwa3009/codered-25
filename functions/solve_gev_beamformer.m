function [w_opt, lambda_max] = solve_gev_beamformer(R_s, R_n, reg_param)
% SOLVE_GEV_BEAMFORMER Solve Generalized Eigenvalue Problem for beamformer weights
%
%   [w_opt, lambda_max] = solve_gev_beamformer(R_s, R_n)
%   [w_opt, lambda_max] = solve_gev_beamformer(R_s, R_n, reg_param)
%
%   Solves the Generalized Eigenvalue Problem (Eq. 9) to find optimal
%   beamformer weights that maximize the signal-to-noise ratio:
%
%       R_s * w = lambda * R_n * w
%
%   The eigenvector corresponding to the largest eigenvalue provides
%   the optimal SNR-maximizing beamformer weights.
%
%   Inputs:
%       R_s       - [num_mics x num_mics] Signal covariance matrix
%       R_n       - [num_mics x num_mics] Noise covariance matrix
%       reg_param - (optional) Regularization parameter. Default: 1e-6
%
%   Outputs:
%       w_opt      - [num_mics x 1] Optimal beamformer weight vector (normalized)
%       lambda_max - Maximum eigenvalue (SNR improvement factor)
%
%   Theory:
%       The GEV solution maximizes the generalized Rayleigh quotient:
%           lambda = (w^H * R_s * w) / (w^H * R_n * w)
%       
%       This represents the ratio of signal power to noise power,
%       making it an optimal criterion for spatial filtering.
%
%   Example:
%       [R_s, R_n] = integrate_spatial_matrix(P, V, [-15, 15]);
%       [w, snr_gain] = solve_gev_beamformer(R_s, R_n);
%       y = w' * x;  % Apply beamformer
%
%   See also: integrate_spatial_matrix, compute_mpdr_power

    % Default regularization
    if nargin < 3
        reg_param = 1e-6;
    end
    
    num_mics = size(R_s, 1);
    
    % Validate inputs
    if size(R_s, 2) ~= num_mics || size(R_n, 1) ~= num_mics || size(R_n, 2) ~= num_mics
        error('R_s and R_n must be square matrices of the same size');
    end
    
    % Add regularization for numerical stability
    R_s_reg = R_s + reg_param * eye(num_mics);
    R_n_reg = R_n + reg_param * eye(num_mics);
    
    % Solve Generalized Eigenvalue Problem
    try
        [V, D] = eig(R_s_reg, R_n_reg);
        
        % Extract eigenvalues (diagonal of D)
        eigenvalues = real(diag(D));
        
        % Find index of maximum eigenvalue
        [lambda_max, max_idx] = max(eigenvalues);
        
        % Extract corresponding eigenvector
        w_opt = V(:, max_idx);
        
    catch ME
        % Fallback: use standard eigenvalue decomposition on R_n^(-1) * R_s
        warning('GEV failed (%s), using fallback method', ME.message);
        
        try
            R_n_inv = inv(R_n_reg);
            M = R_n_inv * R_s_reg;
            
            [V, D] = eig(M);
            eigenvalues = real(diag(D));
            [lambda_max, max_idx] = max(eigenvalues);
            w_opt = V(:, max_idx);
            
        catch
            % Last resort: return uniform weights
            warning('Fallback failed, returning uniform weights');
            w_opt = ones(num_mics, 1) / sqrt(num_mics);
            lambda_max = 1;
        end
    end
    
    % Normalize weight vector
    w_opt = w_opt / norm(w_opt);
    
    % Ensure real eigenvalue (should be real for Hermitian matrices)
    lambda_max = real(lambda_max);
end

