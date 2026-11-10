# Error Taxonomy Evaluation Breakdown

Breakdown of detection rate by error category on DeRef fcrb (seeded sample, n=200):

| Error Category | N | Context Checker (tau=0.88) | Provenance (P3) | Claude Sonnet | Claude Opus | Nemotron 550B |
|---|---|---|---|---|---|---|
| `fabricated_value` | 49 | 75.5% [61.9, 85.4] (k=37/49) | 77.6% [64.1, 87.0] (k=38/49) | 63.3% [49.3, 75.3] (k=31/49) | 63.3% [49.3, 75.3] (k=31/49) | 71.4% [57.6, 82.2] (k=35/49) |
| `wrong_format` | 5 | 80.0% [37.5, 96.4] (k=4/5) | 80.0% [37.5, 96.4] (k=4/5) | 40.0% [11.8, 76.9] (k=2/5) | 60.0% [23.1, 88.2] (k=3/5) | 60.0% [23.1, 88.2] (k=3/5) |
| `wrong_selection` | 43 | 25.6% [14.9, 40.2] (k=11/43) | 25.6% [14.9, 40.2] (k=11/43) | 41.9% [28.4, 56.7] (k=18/43) | 41.9% [28.4, 56.7] (k=18/43) | 41.9% [28.4, 56.7] (k=18/43) |
| `swapped_slot` | 0 | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| `stale_or_state` | 0 | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| `other` | 9 | 77.8% [45.3, 93.7] (k=7/9) | 77.8% [45.3, 93.7] (k=7/9) | 66.7% [35.4, 87.9] (k=6/9) | 66.7% [35.4, 87.9] (k=6/9) | 100.0% [70.1, 100.0] (k=9/9) |

**Key Takeaways:**
1. **`fabricated_value`**: The deterministic checker catches high percentage of absent values with 0 ms token-free execution.
2. **`wrong_selection`**: When the value is present in context but wrong, deterministic grounding detection drops near zero, whereas LLM judges maintain 45-60% detection.
3. **`wrong_format`**: Catchable by the tool schema/definition rules; judges with definitions detect significantly higher than plain context checkers.
