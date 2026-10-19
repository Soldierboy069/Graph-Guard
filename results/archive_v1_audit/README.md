# ARCHIVE: v1 Audit Results (PREVIOUS / INVALID-FOR-FINAL-EVALUATION)

## Notice
The experimental results stored in this archive were generated during the initial Phase 1-12 engineering audit.
They have been retired and marked **INVALID FOR FINAL SCIENTIFIC PRESENTATION** due to two methodological limitations:

1. **Single-Class Test Partition**:
   An arbitrary 70/15/15 chronological split on the first 100,000 Tuesday flows placed the test set in the quiet post-attack window (10:49 to 11:18 AM), resulting in a test partition consisting of 15,000 BENIGN flows and 0 attack flows.
   Evaluating Macro F1 over the single active class yielded an artificial F1 score of 1.0000.

2. **Transductive Label Dependency**:
   The node builder feature `attack_neighbor_ratio` was calculated across the combined dataset using the ground-truth `Label` column, and test edges were present in the global message-passing graph.

## Current Status
These results are preserved solely for audit tracing and historical record.
All active benchmarks are being regenerated under strict inductive temporal evaluation and campaign-window chronological splitting.
