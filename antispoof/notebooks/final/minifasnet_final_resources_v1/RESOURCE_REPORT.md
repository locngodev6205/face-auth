# Frozen MiniFASNet final resources



Official Train/Test populations: 494405 / 67170.

Metadata SHA: fe96b85f575b98d7c231cbf24eb249014b57e95a9f7c1ebd3624034c30d7a30a / 5024238665680fd6df3a1b4b1cf00b75cc60386e6c6978822df86cfe794bd14d.

Sampling: E1 subject overlap removal, subject-disjoint GroupShuffleSplit, stratified exact 100K Train/15K Val; official Test stratified 5K seed 4243. Scale subsets are stratified nested samples.

Counts: {'final_train100k': 100000, 'final_val15k': 15000, 'final_test5k': 5000, 'scale_train20k': 20000, 'scale_val5k': 5000}

Class counts: {'final_train100k': {'1': 46289, '0': 32747, '2': 20964}, 'final_val15k': {'1': 6919, '0': 4919, '2': 3162}, 'final_test5k': {'1': 2642, '0': 1483, '2': 875}, 'scale_train20k': {'1': 9258, '0': 6549, '2': 4193}, 'scale_val5k': {'1': 2306, '0': 1640, '2': 1054}}

SCRFD policy: {'source': 'E1_v5_3', 'model_sha256': '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a', 'insightface_version': '2.0', 'sizes': [[640, 640], [480, 480], [320, 320]], 'det_thresh': 0.5, 'raw_iou_min': 0.5, 'center_shift_max': 0.2, 'fallback': 'official CelebA annotation', 'records': 'face localization only; E1 crop factors removed', 'provider': ['CPUExecutionProvider']}; status counts: {'VALID': 118855, 'FALLBACK_DET_FAIL': 1107, 'FALLBACK_SUSPICIOUS': 38}.

BBox SHA: e15411af09c64581a7567c98e62b97df95bcfde2b3b5affa4af0761f8908ca28. LCC: {'candidate_count': 3766, 'valid_count': 3762, 'class_counts': {'1': 3373, '0': 389}, 'ordered_path_sha256': '7a4946f4bc8c2a31cbf535f804f946fe873b262810af12d3a0ff9a98bf1187af', 'bbox_cache_sha256': 'd6508bde1c7f947ba7c995340c166bde9dee96b77794152a8bb51cbc56fcfbcf', 'detector_sha256': '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a', 'mode': 'verified Stage0 reuse', 'source_manifest_sha256': 'af864ab57bf26af800ceac46c00d9724c8c60f5d0c63037394a1d8708a35abe5', 'source_cache_sha256': '277993f3c8ac436c7fc4f852892a64299b6ccb23063b1774e055a4317c29f78c', 'source_detector_sha256': '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a'}.

Train-only hygiene audit: {'candidate_train_evaluated': 110000, 'status_counts': {'VALID': 108811, 'FALLBACK_DET_FAIL': 1014, 'FALLBACK_SUSPICIOUS': 174, 'INVALID': 1}, 'valid_count': 108811, 'fallback_det_fail_count': 1014, 'fallback_suspicious_count': 174, 'small_face_lt_48_count': 1120, 'suspicious_geometry_count': 174, 'filter_reason_counts': {'TOO_SMALL': 1049, 'SUSPICIOUS_GEOMETRY': 174, 'INVALID_LOCALIZATION': 1}, 'eligible_count': 108776, 'final_sampled_train_count': 100000, 'per_class_candidate_counts': {'0': 36152, '1': 50443, '2': 23405}, 'per_class_filtered_counts': {'0': 531, '1': 92, '2': 601}, 'per_class_final_counts': {'0': 32747, '1': 46289, '2': 20964}, 'overall_filter_rate': 0.011127272727272727, 'per_class_filter_rate': {'0': 0.014687984067271521, '1': 0.0018238407707709692, '2': 0.02567827387310404}, 'class_filter_gap': 0.02385443310233307, 'suspicious_geometry_rate': 0.0015818181818181818, 'guardrails': {'overall_max': 0.05, 'per_class_max': 0.08, 'class_gap_max': 0.03, 'suspicious_max': 0.01}, 'rule': 'TRAIN only: drop suspicious geometry, invalid localization, or bbox minimum side <48 native pixels'}; audit SHA 3ad71b8e56110cbebd34093856f3aa94c6a121f2d2244b3e5a0e50ec8b34a649.

Crop policy: localized face bbox then 2.7x MiniFASNet crop, direct 80x80 BGR float [0,1].

Limitations: subject split approximates E1 group fraction before exact stratified sampling; fallback annotation is retained; LCC only measures external stress.
