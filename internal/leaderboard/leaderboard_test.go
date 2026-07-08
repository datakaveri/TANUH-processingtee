package leaderboard

import (
	"fmt"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
)

// Golden from Python: str(uuid.uuid5(uuid.NAMESPACE_URL, "tanuh:job-test-123"))
func TestUUID5MatchesPython(t *testing.T) {
	got := UUID5("tanuh:job-test-123")
	const golden = "1640b625-aa06-5f8c-a77a-befd07cc4020"
	if got != golden {
		t.Fatalf("uuid5 diverged from Python golden:\n got  %s\n want %s", got, golden)
	}
}

func TestMapMetricsOralCancer(t *testing.T) {
	in := map[string]any{
		"sensitivity": 1.0, "specificity": 0.9, "accuracy": 0.95,
		"ppv": 0.8, "npv": 0.7, "f2": 0.85, "extraneous": 42.0,
	}
	out := MapMetrics(2, in)
	if out["f2_score"] != 0.85 {
		t.Fatalf("f2_score = %v, want 0.85", out["f2_score"])
	}
	if _, ok := out["extraneous"]; ok {
		t.Fatal("extraneous key leaked into oral cancer metrics")
	}
	if _, ok := out["f2"]; ok {
		t.Fatal("raw f2 key should not appear")
	}
}

func TestMapMetricsBreastCancerDerivesWeightedF2(t *testing.T) {
	in := map[string]any{
		"accuracy": 0.9,
		"per_class": map[string]any{
			// support 10, f2(1,1)=1
			"a": map[string]any{"TP": 5.0, "FN": 5.0, "precision": 1.0, "recall": 1.0},
			// support 10, f2(p=0.5,r=0.5): 5*0.25/(2+0.5)=0.5
			"b": map[string]any{"TP": 2.0, "FN": 8.0, "precision": 0.5, "recall": 0.5},
		},
	}
	out := MapMetrics(1, in)
	wf2, ok := out["weighted_f2"].(float64)
	if !ok {
		t.Fatalf("weighted_f2 missing: %v", out)
	}
	if diff := wf2 - 0.75; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("weighted_f2 = %v, want 0.75", wf2)
	}
	if _, ok := out["macro_f2"]; ok {
		t.Fatal("nil macro_f2 should be omitted")
	}
}

func TestMapMetricsUnknownDatasetPassthrough(t *testing.T) {
	in := map[string]any{"anything": 1.0, "nothing": nil}
	out := MapMetrics(3, in)
	if out["anything"] != 1.0 {
		t.Fatal("passthrough failed")
	}
	if _, ok := out["nothing"]; ok {
		t.Fatal("nil value should be omitted")
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		name     string
		err      error
		wantCode int
		wantType string
	}{
		{"user code exit 10", &eval.ExitCodeError{Code: 10}, 1, "EvalUserCodeError"},
		{"cuda exit 11", &eval.ExitCodeError{Code: 11}, 3, "EvalEnvironmentError"},
		{"other exit", &eval.ExitCodeError{Code: 7}, 2, "EvalScriptError"},
		{"env keyword", fmt.Errorf("connection to Secret Manager timed out"), 3, "EnvironmentError"},
		{"deps user error", &eval.DepsError{Output: "no such package: nonexistent-lib"}, 1, "PreprocessingDepsError"},
		{"deps network error", &eval.DepsError{Output: "connection reset while downloading"}, 3, "EnvironmentError"},
		{"generic", fmt.Errorf("something odd happened"), 2, "PipelineError"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := Classify(tc.err)
			if info.Code != tc.wantCode || info.Type != tc.wantType {
				t.Fatalf("got (%d,%s), want (%d,%s)", info.Code, info.Type, tc.wantCode, tc.wantType)
			}
		})
	}
}

func TestSanitizeMsg(t *testing.T) {
	got := SanitizeMsg(`open /app/cvm_workflow/secure_jobs/j1/artifacts/model.onnx: no such file`)
	want := `open <path> no such file`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
