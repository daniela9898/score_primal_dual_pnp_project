# SNORE-PDS experimental package

This package implements an experimental merge of:

1. **Convergent primal-dual PnP-PDS**: primal-dual splitting handles the inverse-problem constraints/data terms.
2. **SNORE stochastic denoising regularization**: the learned prior is used through a stochastic denoising-gradient estimator, by re-noising the current image before applying the denoiser.

The main prototype solves the Gaussian constrained problem

```math
\min_x \lambda R_{\mathrm{SNORE}}(x)
\quad \text{s.t.}\quad \|\Phi x-y\|_2 \le \varepsilon,\; x\in[0,1]^K.
```

The key update is:

```math
x^{k+1} = x^k - \gamma_1\left(\Phi^* w_1^k + w_2^k + \lambda \widehat\nabla R_{\mathrm{SNORE}}(x^k)\right),
```

with the same dual projections as PnP-PDS for the \(\ell_2\)-ball data constraint and the box constraint.

The SNORE gradient estimator implemented by default is

```math
\widehat\nabla R_{\mathrm{SNORE}}(x)
= \frac{x - D_\sigma(x+\sigma\epsilon)}{\sigma^2},
\qquad \epsilon\sim\mathcal N(0,I).
```

This injects Gaussian noise only in the denoiser input. An alternative `residual_mode="noisy"` uses

```math
\frac{x+\sigma\epsilon - D_\sigma(x+\sigma\epsilon)}{\sigma^2}.
```

## Install in your current repo

From the root of `score_primal_dual_pnp_project`:

```bash
unzip /path/to/snore_pds_package.zip -d .
bash scripts/install_snore_pds.sh
```

Then test the CLI:

```bash
python -m snore_pds.run_snore_pds_demo --help
```

## Smoke test

Use any RGB image path:

```bash
python -m snore_pds.run_snore_pds_demo \
  --input data/hundred_images/ILSVRC2012_val_00000491.JPEG.png \
  --task deblur \
  --iters 100 \
  --lambda-snore 1e-5 \
  --gamma1 0.25 \
  --gamma2 0.99 \
  --output results_snore_pds/demo_deblur.png
```

The demo uses a toy Gaussian-smoothing denoiser only to verify the algorithmic plumbing. For experiments, replace it with your real denoiser.

## How to use with your existing denoiser

The solver expects a Python callable with one of these signatures:

```python
denoiser(x_noisy, sigma_float)
denoiser(x_noisy, sigma_tensor)
denoiser(x_noisy)
```

Then call:

```python
from snore_pds import BlurFFT, SNOREConfig, SNOREPDSConfig, snore_pds_gaussian

cfg = SNOREPDSConfig(
    n_iter=300,
    gamma1=0.25,
    gamma2=0.99,
    lambda_snore=1e-5,
    noise_std=2.55/255,
    sigma_begin=5/255,
    sigma_end=2.55/255,
    snore=SNOREConfig(mc_samples=1, residual_mode="clean", clip_denoised=True),
)

x_hat, logs = snore_pds_gaussian(y, op, denoiser, cfg, ground_truth=x_true)
```

## Hyperparameter starting grid

Start very conservative. SNORE gradients scale like `1/sigma^2`, so the regularization weight must be small.

For images in `[0,1]`:

```text
lambda_snore: 1e-6, 3e-6, 1e-5, 3e-5, 1e-4
gamma1:       0.10, 0.20, 0.25, 0.35, 0.45
gamma2:       0.99
sigma_begin:  10/255, 5/255, 2.55/255
sigma_end:    2.55/255 or 1/255
mc_samples:   1 first, then 2 or 4 for ablations
```

Safety check:

```text
gamma1 * gamma2 * (||Phi||^2 + 1) < 1
```

For blur, inpainting, and denoising with normalized operators, `||Phi|| <= 1`, so `gamma1=0.45`, `gamma2=0.99` is close to the PDS limit. For early experiments, use `gamma1=0.20` or `0.25`.

## Recommended first experiments

1. **Denoising sanity check** with `IdentityOperator`.
2. **Deblurring** with the same 7 ImageNet images from your current PDS-PnP experiments.
3. **Random inpainting** using `MaskOperator`.
4. Compare against:
   - attached PnP-PDS baseline,
   - your Score-PDS-PnP baseline,
   - SNORE gradient-descent baseline `snore_pds_l2_unconstrained`.

Log these curves:

- PSNR,
- `data_residual`,
- `relative_update`,
- `snore_grad_norm`,
- `denoise_residual`.

## Important caveat

This is a **research prototype**, not a theorem-complete implementation. The attached PnP-PDS paper has convergence under a firmly nonexpansive denoiser/resolvent setting. Here, SNORE contributes a stochastic nonconvex regularization gradient. So the realistic claim is empirical stability plus possible convergence-to-critical-point style assumptions, not the same monotone-inclusion theorem out of the box.
