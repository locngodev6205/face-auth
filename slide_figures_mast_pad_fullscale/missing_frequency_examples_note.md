# Frequency examples: missing source crops

Figure 7 is not generated. No eligible Real + Attack source crops from CelebA-full are available in the local artifacts inspected. Personal gallery and LFW demo images are not substituted for labeled source PAD samples; CASIA prediction CSVs are not source crops.

Supply one frozen CelebA training Real and one Attack image with valid SCRFD bbox records (or their exact 1.5×/80×80 crops). Use notebook18 `mini_crop_bgr`, `radial_masks` and `spectral_views`, centers [0.0833333333,0.25,0.4166666667], sigma 0.10, gain 0.775, DC preserved. Display BGR input as RGB only for rendering; log FFT magnitude from the clean crop. No identities or filenames on the figure.

Caption: “Frequency counterfactuals are training-only; deployment remains spatial-only.”
