# SNORE-PDS Table V runner

This drop-in version adds:

```bash
python -m snore_pds.run_table5_deblur --help
```

It supports two first experiments:

1. `--method pnp_pds_paper --denoiser paper_dncnn`: paper-style PnP-PDS using the FNE DnCNN checkpoint.
2. `--method snore_pds --denoiser score_pnp`: experimental SNORE-PDS using a score_pnp denoiser/checkpoint.

The runner uses the Gaussian deblurring protocol from Table V: 7 RGB ImageNet images, 128x128 center crop, Gaussian noise `sigma=0.0025`, `alpha=0.82`, and 1200 iterations.

Important: exact reproduction of Table V needs the exact 7 images and `blur_models/blur_1.mat` from the paper repository.
