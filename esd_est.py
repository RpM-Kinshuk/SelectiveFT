import math
import torch
import torch.nn as nn
import numpy as np
import time
from LlaMAft.sampling import *

def net_esd_estimator(
        net=None,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers=None,
        xmin_pos=2,
        conv_norm=0.5, 
        filter_zeros=False,
        use_sliding_window=False,
        num_row_samples=100,  # Required for sliding window
        Q_ratio=2.0,  # Required for sliding window
        step_size=10,  # Sliding window step size for variable ops
        num_sampling_ops_per_dimension=None,  # For fixed number of operations
        eigs_thresh=10  # Minimum number of eigenvalues required for analysis
    ):
    """Estimator for Empirical Spectral Density (ESD) and Alpha parameter.

    Args:
        net (nn.Module): Model to evaluate.
        EVALS_THRESH (float, optional): Threshold to filter near-zero eigenvalues. Defaults to 0.00001.
        bins (int, optional): Number of bins for histogram. Defaults to 100.
        fix_fingers (str, optional): 'xmin_peak' or 'xmin_mid'. Method to select xmin.
        xmin_pos (int, optional): Position in eigenvalue spectrum to choose xmin. Defaults to 2.
        conv_norm (float, optional): Normalization for convolutional layers. Defaults to 0.5.
        filter_zeros (bool, optional): Whether to filter zero eigenvalues. Defaults to False.
        use_sliding_window (bool, optional): Whether to use sliding window sampling. Defaults to False.
        num_row_samples (int, optional): Number of rows to sample in sliding window.
        Q_ratio (float, optional): Ratio of sampled columns to rows in sliding window.
        step_size (int, optional): Step size for sliding window in variable ops mode.
        num_sampling_ops_per_dimension (int, optional): Number of sampling operations for fixed ops mode.
        eigs_thresh (int, optional): Minimum number of eigenvalues required for analysis.

    Returns:
        dict: Results containing spectral norm, alpha values, and other metrics.
    """
    
    results = {
        'alpha': [],
        'spectral_norm': [],
        'D': [],
        'longname': [],
        'eigs': [],
        'norm': [],
        'alphahat': [],
        'stable_rank': [],
        'norm_stable_rank': [],
        'eigs_num': [],
        'layer_time': [],
        'log_nz_eigs_i': [],
        'log_eigs_sum_n': []
    }
    
    tb_start_time = time.time()
    print("======================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print(f"use_sliding_window: {use_sliding_window}, num_row_samples: {num_row_samples}, Q_ratio: {Q_ratio}, step_size: {step_size}, num_sampling_ops_per_dimension: {num_sampling_ops_per_dimension}")
    print("======================================")

    device = next(net.parameters()).device  # type: ignore
    
    for name, m in net.named_modules():  # type: ignore
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if use_sliding_window:
                if isinstance(m, nn.Conv2d):
                    matrix = m.weight.data.clone().to(device)
                    matrix = matrix.float()
                    
                    # Get W matrices for convolutional layer
                    Wmats, N, M, rf = conv2D_Wmats(matrix, channels=CHANNELS.UNKNOWN)
                    rows, cols = Wmats[0].shape
                    
                    results['longname'].append(name)
                    
                    if Q_ratio > 1:
                        num_row_samples_cnn = cols // Q_ratio
                        num_col_samples_cnn = cols
                    else:
                        num_row_samples_cnn = cols
                        num_col_samples_cnn = cols // (1 / Q_ratio)
                    
                    num_row_samples_cnn = max(1, int(num_row_samples_cnn))
                    num_col_samples_cnn = max(1, int(num_col_samples_cnn))
                    
                    temp_results = {
                        'alpha': [],
                        'spectral_norm': [],
                        'D': [],
                        'longname': [],
                        'eigs': [],
                        'norm': [],
                        'alphahat': [],
                        'stable_rank': [],
                        'norm_stable_rank': [],
                        'eigs_num': [],
                        'log_nz_eigs_i': [],
                        'log_eigs_sum_n': []
                    }
                    
                    for i in range(int(rows // cols)):
                        eig_values = []
                        
                        for W in Wmats:
                            submatrix = W[i*num_row_samples_cnn:(i+1)*num_row_samples_cnn, :num_col_samples_cnn]
                            submatrix *= math.sqrt(conv_norm)
                            eig_values.append(torch.square(torch.linalg.svdvals(submatrix)))
                        
                        eig_values = torch.cat(eig_values)
                        eig_values = torch.sort(eig_values, descending=False).values
                        
                        analyze_esd(
                            results=temp_results,
                            eigs=eig_values,
                            EVALS_THRESH=EVALS_THRESH,
                            bins=bins,
                            fix_fingers=fix_fingers,
                            xmin_pos=xmin_pos,
                            filter_zeros=filter_zeros,
                            device=device
                        )
                    
                    # Average all the results from different sliding windows
                    results['alpha'].append(np.mean(temp_results['alpha']))
                    results['spectral_norm'].append(np.mean(temp_results['spectral_norm']))
                    results['D'].append(np.mean(temp_results['D']))
                    results['norm'].append(np.mean(temp_results['norm']))
                    results['alphahat'].append(np.mean(temp_results['alphahat']))
                    results['stable_rank'].append(np.mean(temp_results['stable_rank']))
                    results['norm_stable_rank'].append(np.mean(temp_results['norm_stable_rank']))
                    results['eigs'].append(temp_results['eigs'])
                    results['eigs_num'].append(np.sum(temp_results['eigs_num']))
                    results['log_nz_eigs_i'].append(np.mean(temp_results['log_nz_eigs_i']))
                    results['log_eigs_sum_n'].append(np.mean(temp_results['log_eigs_sum_n']))
                    
                elif isinstance(m, nn.Linear):
                    matrix = m.weight.data.clone().to(device)
                    matrix = matrix.float()
                    
                    eigs = sampled_eigs(
                        matrix=matrix, 
                        isconv2d=isinstance(m, nn.Conv2d),
                        conv_norm=conv_norm, 
                        num_row_samples=num_row_samples,
                        Q_ratio=Q_ratio, 
                        step_size=step_size,
                        sampling_ops_per_dim=num_sampling_ops_per_dimension
                    )
                    
                    if len(eigs) < eigs_thresh:
                        continue
                    
                    results['longname'].append(name)
                    analyze_esd(
                        results=results,
                        eigs=eigs,
                        EVALS_THRESH=EVALS_THRESH,
                        bins=bins,
                        fix_fingers=fix_fingers,
                        xmin_pos=xmin_pos,
                        filter_zeros=filter_zeros,
                        device=device
                    )
            else:
                # Regular full matrix ESD computation
                matrix = m.weight.data.clone().to(device)
                matrix = matrix.float()
                
                if isinstance(m, nn.Conv2d):
                    matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                    matrix = matrix.transpose(1, 2).transpose(0, 1)
                
                # Calculate eigenvalues using SVD
                eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
                
                if len(eigs) < eigs_thresh:
                    continue
                
                results['longname'].append(name)
                analyze_esd(
                    results=results,
                    eigs=eigs,
                    EVALS_THRESH=EVALS_THRESH,
                    bins=bins,
                    fix_fingers=fix_fingers,
                    xmin_pos=xmin_pos,
                    filter_zeros=filter_zeros,
                    device=device
                )
    
    tb_end_time = time.time()
    results['layer_time'].append(tb_end_time - tb_start_time)
    
    return results

def analyze_esd(results, eigs, EVALS_THRESH, bins, fix_fingers, xmin_pos, filter_zeros, device):
    """Analyze eigenvalue spectrum and compute power law statistics.
    
    Args:
        results: Dictionary to store results
        eigs: Eigenvalues to analyze
        EVALS_THRESH: Threshold for filtering small eigenvalues
        bins: Number of histogram bins
        fix_fingers: Method for selecting xmin ('xmin_peak' or 'xmin_mid')
        xmin_pos: Position in spectrum to select xmin
        filter_zeros: Whether to filter near-zero eigenvalues
        device: Computation device
        
    Returns:
        Updated results dictionary
    """
    # Sort eigenvalues in ascending order
    eigs = torch.sort(eigs, descending=False).values
    
    # Calculate spectral norm and Frobenius norm
    spectral_norm = eigs[-1].item()
    fnorm = torch.sum(eigs).item()
    
    # Stable rank calculation
    stable_rank = fnorm / spectral_norm
    norm_stable_rank = stable_rank / len(eigs)
    
    # Filter based on threshold
    nz_eigs = eigs[eigs > EVALS_THRESH] if filter_zeros else eigs
    if len(nz_eigs) == 0:
        nz_eigs = eigs
    
    N = len(nz_eigs)
    log_nz_eigs = torch.log(nz_eigs)
    
    # Alpha and D calculations
    if fix_fingers == 'xmin_mid':
        i = N // xmin_pos
        xmin = nz_eigs[i]
        n = float(N - i)
        seq = torch.arange(n, device=device)
        final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n))
        
        # Store additional diagnostics
        log_nz_eigs_i = log_nz_eigs[i].item()
        log_eigs_sum_n = (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i]) / n
    else:
        alphas = torch.zeros(N-1, device=device)
        Ds = torch.ones(N-1, device=device)
        
        if fix_fingers == 'xmin_peak':
            hist_nz_eigs = torch.log10(nz_eigs)
            min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
            counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
            boundaries = torch.linspace(min_e, max_e, bins + 1)
            ih = torch.argmax(counts)
            xmin2 = 10 ** boundaries[ih]
            xmin_min = torch.log10(0.95 * xmin2)
            xmin_max = 1.5 * xmin2
        
        log_nz_eigs_i = 0
        log_eigs_sum_n = 0
        
        for i, xmin in enumerate(nz_eigs[:-1]):
            if fix_fingers == 'xmin_peak':
                if xmin < xmin_min:
                    continue
                if xmin > xmin_max:
                    break
            
            n = float(N - i)
            seq = torch.arange(n, device=device)
            alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            alphas[i] = alpha
            
            if alpha > 1:
                Ds[i] = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))
        
        min_D_index = torch.argmin(Ds)
        final_alpha = alphas[min_D_index]
        final_D = Ds[min_D_index]
        
        # For the best alpha, calculate diagnostics
        i = min_D_index
        if i >= 0 and i < len(nz_eigs):
            n = float(N - i)
            log_nz_eigs_i = log_nz_eigs[i].item()
            log_eigs_sum_n = (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i]) / n
    
    # Store results
    final_alpha = final_alpha.item()
    final_D = final_D.item()
    final_alphahat = final_alpha * math.log10(spectral_norm)
    
    results['spectral_norm'].append(spectral_norm)
    results['alphahat'].append(final_alphahat)
    results['norm'].append(fnorm)
    results['alpha'].append(final_alpha)
    results['D'].append(final_D)
    results['eigs'].append(nz_eigs.cpu().numpy())
    results['eigs_num'].append(len(nz_eigs))
    results['stable_rank'].append(stable_rank)
    results['norm_stable_rank'].append(norm_stable_rank)
    results['log_nz_eigs_i'].append(log_nz_eigs_i)
    results['log_eigs_sum_n'].append(log_eigs_sum_n)
    
    return results