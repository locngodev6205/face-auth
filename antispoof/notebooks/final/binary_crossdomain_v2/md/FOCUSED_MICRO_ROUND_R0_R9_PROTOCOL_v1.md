# Focused micro R0–R9 round, protocol v1

## Question and comparisons

This is a single-seed micro development round around P3. The treatments are clean CE (R0); adaptive worst-view CE weights 0.15, 0.25, 0.35, 0.50 (R2/R1/R3/R4); all-three CE (R6); harmful-gated worst CE (R7); worst/random CE mix (R8); and worst CE plus KL 0.02 (R9). R5 is intentionally absent. The aim is to measure how these mechanisms change source and cross-domain behavior, not to presume a winner.

All runs add `L_aux = 0.1 L_spoof_type + 0.1 L_lighting + L_attributes` on the clean view; attributes apply to Real samples only.

| Run | PAD loss, before adding `L_aux` |
|---|---|
| R0_clean | `CE_clean` |
| R1_p3_ce25 | `0.75 CE_clean + 0.25 CE_worst` |
| R2_worst_ce15 | `0.85 CE_clean + 0.15 CE_worst` |
| R3_worst_ce35 | `0.65 CE_clean + 0.35 CE_worst` |
| R4_worst_ce50 | `0.50 CE_clean + 0.50 CE_worst` |
| R6_all3_ce25 | `0.75 CE_clean + 0.25 mean(CE_LOW, CE_MID, CE_HIGH)` |
| R7_harmful_gated_worst | `0.75 CE_clean + 0.25 CE_harmful`; detached gate is `m_worst < m_clean` |
| R8_worst_random_mix | `0.75 CE_clean + 0.25 (0.75 CE_worst + 0.25 CE_random)` |
| R9_worst_ce25_kl002 | `0.75 CE_clean + 0.25 CE_worst + 0.02 KL(softmax(z_clean.detach()) || softmax(z_worst))` |

## Shared controls

Both notebooks use the frozen `mini_train10k.csv` and `mini_val3k.csv` from the completed 00d resource bundle, with train seed 100, the same official PAD-pretrained MiniFASNetV2 initialization, sample order, bbox jitter, Albumentations, batch 256, SGD (lr 0.005, momentum 0.9, weight decay 5e-4), at most 20 epochs, MultiStepLR milestones [6,14] and gamma 0.2. The source Val3K minimum ACER selects checkpoints, with AUC tie-break. Spectral runs warm up for one clean epoch; that epoch cannot become their best checkpoint or count toward patience. Early stopping starts no earlier than epoch 8 with patience 4. R0 can select from epoch 1.

Spectral bands use radial centers [0.15,0.45,0.75], sigma 0.10, gain Uniform(0.65,0.90), preserved DC and unchanged phase. Worst selection runs in eval/no_grad mode to protect batch normalization; optimization forwards remain differentiable. Bbox jitter, base augmentation, spectral gain, and random-band selection use separate deterministic RNG streams. The only intended treatment is the spectral loss mechanism.

## Frozen evaluation

After training, reload `best.pth`, recompute and freeze the Val3K threshold, and then evaluate LCC training, development, official evaluation, and combined. `EVAL_CELEBA_TEST=False` skips CelebA full Test inference by default during micro screening; set it to `True` only when that optional evaluation is needed. CelebA full Test will be evaluated later for selected scale-up or final candidates. LCC is an external-development domain and may later inform method selection; it is not an untouched final benchmark. CASIA-FASD is not used. No LCC threshold fitting or detector inference occurs. Combined LCC concatenates three physical-split prediction tables and recomputes metrics from their scored rows. Report AUC, EER, TPR@FPR1%, APCER, BPCER, ACER/HTER, accuracy, candidate count, scored count, coverage, and the locked threshold.

Each run saves its own config, best/last checkpoint, history, frozen threshold, metrics, predictions, and diagnostics. Resume states include optimizer, scheduler, scaler, patience, history, and RNG states. The two notebooks run independently; neither imports the other's results. Descriptive source guard: Val ACER <= R0 Val ACER + 0.005 when R0 is available. A guard failure is reported without deleting or rerunning a model. No automatic winner or new search is launched.

Validate notebook parsing, code-cell compilation, run IDs, loss gradients on tiny synthetic batches, strict pretrained loading, and absence of target threshold fitting before Kaggle training. Each selected run costs up to 20 Train10K epochs plus Val3K per epoch, then three physical LCC evaluations; optional CelebA full Test inference adds cost only when enabled. The single seed limits statistical claims; GPU training and raw-image evaluation must supply measured results before any mechanism is preferred.
