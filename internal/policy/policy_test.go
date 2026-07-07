package policy

import "testing"

// TestHashMatchesPython pins the canonical hash to the value produced by the
// original Python implementation (json.dumps(policy, sort_keys=True,
// separators=(",", ":")) + sha256) for the shipped network_policy.json.
// If this test fails after editing the policy file, recompute the golden:
//
//	python3 -c "import hashlib,json; print(hashlib.sha256(json.dumps(json.load(open('policy/network_policy.json')),sort_keys=True,separators=(',',':')).encode()).hexdigest())"
func TestHashMatchesPython(t *testing.T) {
	p, err := Load("../../policy/network_policy.json")
	if err != nil {
		t.Fatalf("load policy: %v", err)
	}
	got, err := Hash(p)
	if err != nil {
		t.Fatalf("hash policy: %v", err)
	}
	const golden = "b0d0f1fa45ba434c50b99c77c5d3f27ee873ccb553319f54b16eb551d852ba93"
	if got != golden {
		t.Fatalf("policy hash diverged from Python golden:\n got  %s\n want %s", got, golden)
	}
}

func TestCanonicalJSONSortsAndCompacts(t *testing.T) {
	v := map[string]any{"b": 2.0, "a": []any{map[string]any{"y": 1.0, "x": "s"}}}
	got, err := CanonicalJSON(v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"a":[{"x":"s","y":1}],"b":2}`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
