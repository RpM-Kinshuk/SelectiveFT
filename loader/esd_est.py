import torch
import torch.nn as nn
import math

def net_esd_estimator(
        net=None,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers=None,
        xmin_pos=2,
        conv_norm=0.5, 
        filter_zeros=False):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
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
    print("=================================")
    
    device = next(net.parameters()).device  # type: ignore

    for name, m in net.named_modules(): # type: ignore
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone().to(device)
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            matrix = matrix.float()
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            eigs = torch.sort(eigs).values
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            
            nz_eigs = eigs[eigs > EVALS_THRESH] if filter_zeros else eigs
            if len(nz_eigs) == 0:
                nz_eigs = eigs
            N = len(nz_eigs)
            log_nz_eigs = torch.log(nz_eigs)

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
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()
            final_alphahat = final_alpha * math.log10(spectral_norm)

            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.cpu().numpy())

    return results