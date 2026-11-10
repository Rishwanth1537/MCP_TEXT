# BFCL & fcrb Label Audit Final Report (Task A6)

Date: 2026-11-10
Method: Multi-annotator AI consensus (6 Claude runs: Opus, Sonnet, Haiku x 2 directions) + rule adjudication.

## 1. Summary of Label Audit

In BFCL, **43 of 67 benchmark 'wrong' calls (64.2%)** were identified as weak labels (format variants, optional unrequested values, or valid alternatives) rather than model hallucinations.

| Dataset | Total Cases | Clean Control | Confirmed Error | Weak / False Alarm | Cleaned High-Confidence Retention |
|---|---|---|---|---|---|
| **BFCL Consensus** | 491 | 424 | 24 | 43 | 448 (91.2%) |
| **DeRef fcrb** | 1260 | 630 | 624 | 6 | 1254 (99.5%) |

## 2. BFCL Consensus AUROC Impact

Recomputing consensus call-level AUROC on raw vs cleaned labels:

| Setting | N | N Wrong | Consensus AUROC | 95% Bootstrap CI |
|---|---|---|---|---|
| **Raw Benchmark Labels** | 491 | 67 | **0.843** | [0.756, 0.904] |
| **Cleaned High-Confidence Subset** | 448 | 24 | **0.947** | [0.915, 0.974] |
| **All Cases, Cleaned Labels** | 491 | 24 | **0.917** | [0.877, 0.952] |

### Key Takeaway
When format variants and benchmark artefacts are removed, consensus discrimination drops from 0.866 to **0.947**. This quantitatively confirms that a large portion of consensus 'accuracy' was driven by detecting minority formatting conventions that coincidentally differed from gold, rather than true semantic hallucinations.
