import math
import torch
import torch.nn as nn
from loader.sampling import *

def net_esd_estimator(
        net=None,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers='xmin_mid',
        xmin_pos=2,
        conv_norm=0.5, 
        filter_zeros=False,
        use_sliding_window=False,
        num_row_samples=100,  # Required for sliding window
        Q_ratio=2.0,  # Required for sliding window
        step_size=10,  # Sliding window step size for variable ops
        num_sampling_ops_per_dimension=None  # For fixed number of operations
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
        'alphahat': []
    }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print(f"use_sliding_window: {use_sliding_window}, num_row_samples: {num_row_samples}, Q_ratio: {Q_ratio}, step_size: {step_size}, num_sampling_ops_per_dimension: {num_sampling_ops_per_dimension}")
    print("=================================")

    device = next(net.parameters()).device  # type: ignore

    for name, m in net.named_modules():  # type: ignore
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone().to(device)
            matrix = matrix.float()

            # Sliding window option for sampling ESD
            if use_sliding_window:
                eigs = sampled_eigs(
                        matrix=matrix, isconv2d=isinstance(m, nn.Conv2d),
                        conv_norm=conv_norm, num_row_samples=num_row_samples,
                        Q_ratio=Q_ratio, step_size=step_size,
                        sampling_ops_per_dim=num_sampling_ops_per_dimension
                    )
            else:
                # Regular full matrix ESD computation
                eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            
            if not isinstance(eigs, torch.Tensor):
                eigs = torch.tensor(eigs, device=device)
                
            eigs = torch.sort(eigs).values
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            
            # Filter based on threshold
            nz_eigs = eigs[eigs > EVALS_THRESH] if filter_zeros else eigs
            if len(nz_eigs) == 0:
                nz_eigs = eigs
            N = len(nz_eigs)
            log_nz_eigs = torch.log(nz_eigs)
            
            # Alpha and D calculations (from before)
            if fix_fingers == 'xmin_mid':
                i = N // xmin_pos
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n, device=device)
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n))
            else:
                alphas = torch.zeros(N-1, device=device)
                Ds = torch.ones(N-1, device=device)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e) # type: ignore
                    boundaries = torch.linspace(min_e, max_e, bins + 1) # type: ignore
                    ih = torch.argmax(counts)
                    xmin2 = 10 ** boundaries[ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
               
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
            
            # Store results
            final_alpha = final_alpha.item()  # type: ignore
            final_D = final_D.item()  # type: ignore
            final_alphahat = final_alpha * math.log10(spectral_norm)
            
            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(nz_eigs.cpu().numpy())
            results['eigs_num'].append(len(nz_eigs))
    
    return results