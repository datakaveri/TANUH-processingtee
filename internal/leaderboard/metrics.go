package leaderboard

// datasetSlug maps internal dataset_id → leaderboard vertical slug.
var datasetSlug = map[int]string{1: "breast_cancer", 2: "oral_cancer", 3: "glaucoma"}

// f2FromPrecisionRecall computes F-beta=2 from precision/recall:
// 5pr / (4p + r). 0 when undefined.
func f2FromPrecisionRecall(p, r float64) float64 {
	denom := 4.0*p + r
	if denom == 0 {
		return 0
	}
	return 5.0 * p * r / denom
}

// MapMetrics maps the internal metrics dict onto the leaderboard's
// per-vertical schema. Fields we can't produce are omitted rather than
// guessed. Ported 1:1 from _leaderboard_metrics.
func MapMetrics(datasetID int, metrics map[string]any) map[string]any {
	m := metrics
	if m == nil {
		m = map[string]any{}
	}

	var out map[string]any
	switch datasetID {
	case 2: // oral_cancer → OralCancerMetrics
		out = map[string]any{
			"sensitivity": m["sensitivity"],
			"specificity": m["specificity"],
			"accuracy":    m["accuracy"],
			"ppv":         m["ppv"],
			"npv":         m["npv"],
			"f2_score":    m["f2"],
		}
	case 1: // breast_cancer → BreastCancerMetrics
		// weighted_f2 isn't emitted directly; derive it from per-class
		// precision/recall weighted by class support when available.
		weightedF2 := m["weighted_f2"]
		perClass, _ := m["per_class"].(map[string]any)
		if weightedF2 == nil && len(perClass) > 0 {
			total := 0.0
			acc := 0.0
			for _, v := range perClass {
				c, _ := v.(map[string]any)
				support := toFloat(c["TP"]) + toFloat(c["FN"])
				total += support
				acc += support * f2FromPrecisionRecall(toFloat(c["precision"]), toFloat(c["recall"]))
			}
			if total > 0 {
				weightedF2 = acc / total
			}
		}
		out = map[string]any{
			"accuracy":         m["accuracy"],
			"macro_f2":         m["macro_f2"],
			"weighted_f2":      weightedF2,
			"macro_f1":         m["macro_f1"],
			"sensitivity":      m["macro_recall"],
			"qwk":              m["qwk"],
			"specificity":      m["macro_specificity"],
			"npv":              m["macro_npv"],
			"ppv":              m["macro_ppv"],
			"confusion_matrix": m["confusion_matrix"],
			"auc":              m["auc"],
		}
	default:
		out = make(map[string]any, len(m))
		for k, v := range m {
			out[k] = v
		}
	}

	for k, v := range out {
		if v == nil {
			delete(out, k)
		}
	}
	return out
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	}
	return 0
}
