# Frequency counterfactual examples: source images needed

Not generated: `fig_frequency_counterfactual_examples.png/.pdf`.

The repository contains run metrics, CSV manifests and bbox caches, but the 100K source JPEGs are not present in the inspected workspace. Existing EDA galleries and plots cannot establish the original crop geometry / sample membership and were not substituted for dataset images.

To complete this figure, mount the original CelebA-Spoof images (the run config uses a Kaggle-only `celeba_root`), choose one Real and one Attack entry from the frozen Train100K manifest, and resolve their valid bbox-cache records. Use `mini_crop_bgr`, `radial_masks` and `spectral_views` directly from notebook 17. Do not resample the training manifest or alter cached bboxes.

Frozen rendering settings: crop1.5; 80×80 BGR float [0,1]; normalized radius sqrt(x²+y²)/sqrt(2×40²); centers [0.0833333333, 0.25, 0.4166666667]; sigma 0.10; gain 0.775; preserve DC; keep phase; clamp inverse-FFT output to [0,1]. Convert BGR to RGB only for display. Display clean log FFT magnitude as the fifth column. No subject names, IDs or paths in the figure.

Caption: “Frequency views are used only during training; inference uses the clean spatial model.”
