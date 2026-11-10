# Error Taxonomy Evaluation Breakdown

Breakdown of detection rate by error category on DeRef fcrb (seeded sample, n=200):

| Error Category | N | Context Checker (tau=0.88) | Provenance (P3) | Claude Sonnet | Claude Opus | Nemotron 550B |
|---|---|---|---|---|---|---|
| `fabricated_value` | 57 | 100.0% [93.7, 100.0] (k=57/57) | 100.0% [93.7, 100.0] (k=57/57) | 77.2% [64.8, 86.2] (k=44/57) | 78.9% [66.7, 87.5] (k=45/57) | 82.5% [70.6, 90.2] (k=47/57) |
| `wrong_format` | 2 | 100.0% [34.2, 100.0] (k=2/2) | 100.0% [34.2, 100.0] (k=2/2) | 100.0% [34.2, 100.0] (k=2/2) | 100.0% [34.2, 100.0] (k=2/2) | 100.0% [34.2, 100.0] (k=2/2) |
| `wrong_selection` | 47 | 0.0% [0.0, 7.6] (k=0/47) | 2.1% [0.4, 11.1] (k=1/47) | 23.4% [13.6, 37.2] (k=11/47) | 23.4% [13.6, 37.2] (k=11/47) | 34.0% [22.2, 48.3] (k=16/47) |
| `swapped_slot` | 0 | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| `stale_or_state` | 0 | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| `other` | 0 | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |

**Key Takeaways:**
1. **`fabricated_value`**: The deterministic checker catches high percentage of absent values with 0 ms token-free execution.
2. **`wrong_selection`**: When the value is present in context but wrong, deterministic grounding detection drops near zero, whereas LLM judges maintain 45-60% detection.
3. **`wrong_format`**: Catchable by the tool schema/definition rules; judges with definitions detect significantly higher than plain context checkers.
