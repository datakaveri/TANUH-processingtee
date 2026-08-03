// Package integration is a single, standalone test file that exercises the
// Processing TEE end to end across package boundaries, using only each
// package's exported surface.
package systemsTest

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
	"github.com/datakaveri/tanuh-processing-tee/internal/leaderboard"
	"github.com/datakaveri/tanuh-processing-tee/internal/pipeline"
	"github.com/datakaveri/tanuh-processing-tee/internal/policy"
	"github.com/datakaveri/tanuh-processing-tee/internal/ratls"
	"github.com/datakaveri/tanuh-processing-tee/internal/server"
)

// ---------------------------------------------------------------------
// 1. config — environment parsing is the root of every other component.
// ---------------------------------------------------------------------

func TestConfig_FromEnvDefaultsAndOverrides(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		clearProcessingTeeEnv(t)
		cfg := config.FromEnv()

		if cfg.ListenAddr != ":443" {
			t.Errorf("ListenAddr = %q, want :443", cfg.ListenAddr)
		}
		if cfg.BaseDir != "/app" {
			t.Errorf("BaseDir = %q, want /app", cfg.BaseDir)
		}
		if got, want := cfg.WorkflowDir(), "/app/cvm_workflow"; got != want {
			t.Errorf("WorkflowDir() = %q, want %q", got, want)
		}
		if cfg.IdleTimeout != 300*time.Second {
			t.Errorf("IdleTimeout = %s, want 300s", cfg.IdleTimeout)
		}
		if !cfg.DeallocAfterJob {
			t.Error("DeallocAfterJob should default true (PROCESSING_DEALLOCATE_AFTER_JOB defaults to \"1\")")
		}
		if len(cfg.Datasets) != 2 {
			t.Fatalf("expected 2 built-in datasets, got %d", len(cfg.Datasets))
		}
		ds, err := cfg.Dataset(1)
		if err != nil || ds.SecretID != "tanuh-ds1-key" {
			t.Errorf("Dataset(1) = %+v, err=%v", ds, err)
		}
	})

	t.Run("overrides", func(t *testing.T) {
		clearProcessingTeeEnv(t)
		t.Setenv("BASE_DIR", "/tmp/custom-base")
		t.Setenv("LISTEN_ADDR", ":9443")
		t.Setenv("PROCESSING_DEALLOCATE_AFTER_JOB", "0")
		t.Setenv("PROCESSING_IDLE_TIMEOUT_SECONDS", "45")

		cfg := config.FromEnv()
		if cfg.ListenAddr != ":9443" {
			t.Errorf("ListenAddr = %q, want :9443", cfg.ListenAddr)
		}
		if cfg.DeallocAfterJob {
			t.Error("DeallocAfterJob should be false when PROCESSING_DEALLOCATE_AFTER_JOB=0")
		}
		if cfg.IdleTimeout != 45*time.Second {
			t.Errorf("IdleTimeout = %s, want 45s", cfg.IdleTimeout)
		}
		if got, want := cfg.WorkflowDir(), "/tmp/custom-base/cvm_workflow"; got != want {
			t.Errorf("WorkflowDir() = %q, want %q", got, want)
		}
	})

	t.Run("unknown dataset id errors", func(t *testing.T) {
		clearProcessingTeeEnv(t)
		cfg := config.FromEnv()
		if _, err := cfg.Dataset(99); err == nil {
			t.Error("Dataset(99) should error for an unconfigured id")
		}
	})
}

func clearProcessingTeeEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"BASE_DIR", "LISTEN_ADDR", "RATLS_AUDIENCE", "GCP_PROJECT_ID",
		"DATASETS_BUCKET", "EVAL_SCRIPTS_BUCKET", "PROJECT", "ZONE", "INSTANCE",
		"PROCESSING_IDLE_TIMEOUT_SECONDS", "PROCESSING_DEALLOCATE_AFTER_JOB",
		"PROCESSING_EVAL_TIMEOUT_SECONDS", "PROCESSING_DEPS_TIMEOUT_SECONDS",
		"LEADERBOARD_SUBMIT_URL", "NETWORK_POLICY_PATH", "CALLBACK_AUDIENCE",
	} {
		t.Setenv(k, "")
		os.Unsetenv(k) //nolint:errcheck
	}
}

// ---------------------------------------------------------------------
// 2. crypto — the dataset confidentiality boundary. Integration of /internal/crypto/aesgcm_test.go
// ---------------------------------------------------------------------

func TestCrypto_DatasetArtifactDecryption(t *testing.T) {
	key := bytes.Repeat([]byte{0x42}, 32)

	t.Run("blob roundtrip and fail-closed on tamper", func(t *testing.T) {
		plaintext := []byte(`{"images":[{"path":"a.png","label":1}]}`)
		enc := sealBlob(t, key, plaintext)

		got, err := crypto.DecryptBlob(key, enc)
		if err != nil {
			t.Fatalf("DecryptBlob: %v", err)
		}
		if !bytes.Equal(got, plaintext) {
			t.Fatalf("roundtrip mismatch: got %q", got)
		}

		tampered := append([]byte(nil), enc...)
		tampered[len(tampered)-1] ^= 0xFF
		if _, err := crypto.DecryptBlob(key, tampered); err == nil {
			t.Fatal("tampered ciphertext must not decrypt (integrity is the pipeline's only guard)")
		}
	})

	t.Run("blob rejects malformed wire format", func(t *testing.T) {
		if _, err := crypto.DecryptBlob(key, []byte{0, 0}); err == nil {
			t.Error("blob shorter than the 4-byte length header must error")
		}
		if _, err := crypto.DecryptBlob(key, []byte{0xFF, 0xFF, 0xFF, 0xFF, 1, 2}); err == nil {
			t.Error("a nonce_len larger than the blob must error, not panic")
		}
	})

	t.Run("chunked file roundtrip across chunk-size edge cases", func(t *testing.T) {
		dir := t.TempDir()
		chunks := [][]byte{
			bytes.Repeat([]byte("image-bytes-"), 500), // an ordinary chunk
			{},          // an empty final chunk (still GCM-tagged)
			[]byte("x"), // a single-byte chunk
		}
		encPath := filepath.Join(dir, "images.zip.enc")
		writeChunkedFile(t, encPath, key, chunks)

		outPath := filepath.Join(dir, "images.zip")
		if err := crypto.DecryptChunkedFile(key, encPath, outPath); err != nil {
			t.Fatalf("DecryptChunkedFile: %v", err)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatal(err)
		}
		if want := bytes.Join(chunks, nil); !bytes.Equal(got, want) {
			t.Fatalf("chunked roundtrip mismatch: got %d bytes, want %d", len(got), len(want))
		}
	})

	t.Run("chunked file rejects a corrupted middle chunk", func(t *testing.T) {
		dir := t.TempDir()
		chunks := [][]byte{bytes.Repeat([]byte("A"), 100), bytes.Repeat([]byte("B"), 100)}
		encPath := filepath.Join(dir, "in.enc")
		writeChunkedFile(t, encPath, key, chunks)

		raw, err := os.ReadFile(encPath)
		if err != nil {
			t.Fatal(err)
		}
		raw[len(raw)-1] ^= 0xFF // flip a byte inside the last chunk's GCM tag
		if err := os.WriteFile(encPath, raw, 0o644); err != nil {
			t.Fatal(err)
		}

		if err := crypto.DecryptChunkedFile(key, encPath, filepath.Join(dir, "out.bin")); err == nil {
			t.Fatal("corrupted chunk must fail authentication, not decrypt silently")
		}
	})
}

func sealBlob(t *testing.T, key, plaintext []byte) []byte {
	t.Helper()
	block, err := aes.NewCipher(key)
	if err != nil {
		t.Fatal(err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		t.Fatal(err)
	}
	nonce := make([]byte, 12)
	if _, err := rand.Read(nonce); err != nil {
		t.Fatal(err)
	}
	ct := aead.Seal(nil, nonce, plaintext, nil)

	var buf bytes.Buffer
	binary.Write(&buf, binary.BigEndian, uint32(len(nonce))) //nolint:errcheck
	buf.Write(nonce)
	buf.Write(ct)
	return buf.Bytes()
}

func writeChunkedFile(t *testing.T, path string, key []byte, chunks [][]byte) {
	t.Helper()
	block, err := aes.NewCipher(key)
	if err != nil {
		t.Fatal(err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	binary.Write(&buf, binary.BigEndian, uint32(len(chunks))) //nolint:errcheck
	for _, pt := range chunks {
		nonce := make([]byte, 12)
		if _, err := rand.Read(nonce); err != nil {
			t.Fatal(err)
		}
		ct := aead.Seal(nil, nonce, pt, nil)
		binary.Write(&buf, binary.BigEndian, uint32(len(nonce)+len(ct))) //nolint:errcheck
		buf.Write(nonce)
		buf.Write(ct)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------
// 3. eval — the Go/Python subprocess contract (frozen exit codes).
// ---------------------------------------------------------------------

func TestEval_SubprocessContract(t *testing.T) {
	fakePython := buildFakePython(t)
	restorePath := prependPATH(t, filepath.Dir(fakePython))
	defer restorePath()

	dir := t.TempDir()
	script := filepath.Join(dir, "evaluation_script_1.py")
	if err := os.WriteFile(script, []byte("#stub\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "model.onnx")
	dataset := filepath.Join(dir, "dataset.json")
	results := filepath.Join(dir, "results.json")
	for _, p := range []string{model, dataset} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	cases := []struct {
		name       string
		exit       string
		wantNil    bool
		wantCode   int
		wantIsExit bool
	}{
		{"success", "0", true, 0, false},
		{"user preprocessing failure", "10", false, 10, true},
		{"cuda/gpu environment failure", "11", false, 11, true},
		{"unclassified script error", "7", false, 7, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAKE_PYTHON_EXIT", tc.exit)
			t.Setenv("FAKE_PYTHON_SLEEP", "")
			err := eval.Run(context.Background(), dir, script, model, dataset, results, "", 0)
			if tc.wantNil {
				if err != nil {
					t.Fatalf("Run() = %v, want nil", err)
				}
				return
			}
			var exitErr *eval.ExitCodeError
			if tc.wantIsExit && !errors.As(err, &exitErr) {
				t.Fatalf("Run() = %v (%T), want *eval.ExitCodeError", err, err)
			}
			if exitErr.Code != tc.wantCode {
				t.Fatalf("exit code = %d, want %d", exitErr.Code, tc.wantCode)
			}
		})
	}

	t.Run("preprocessing flag only appended when the file actually exists", func(t *testing.T) {
		t.Setenv("FAKE_PYTHON_EXIT", "0")
		t.Setenv("FAKE_PYTHON_SLEEP", "")
		argvFile := filepath.Join(dir, "argv.txt")
		t.Setenv("FAKE_PYTHON_ARGV_FILE", argvFile)
		defer os.Unsetenv("FAKE_PYTHON_ARGV_FILE") //nolint:errcheck

		missing := filepath.Join(dir, "does-not-exist.py")
		if err := eval.Run(context.Background(), dir, script, model, dataset, results, missing, 0); err != nil {
			t.Fatalf("Run() with missing preprocessing path: %v", err)
		}
		argv, _ := os.ReadFile(argvFile)
		if strings.Contains(string(argv), "--preprocessing") {
			t.Errorf("argv should omit --preprocessing when the file is absent, got %q", argv)
		}

		present := filepath.Join(dir, "preprocessing.py")
		if err := os.WriteFile(present, []byte("#"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := eval.Run(context.Background(), dir, script, model, dataset, results, present, 0); err != nil {
			t.Fatalf("Run() with present preprocessing path: %v", err)
		}
		argv, _ = os.ReadFile(argvFile)
		if !strings.Contains(string(argv), "--preprocessing") {
			t.Errorf("argv should include --preprocessing when the file exists, got %q", argv)
		}
	})

	t.Run("hard timeout cap is enforced", func(t *testing.T) {
		t.Setenv("FAKE_PYTHON_EXIT", "0")
		t.Setenv("FAKE_PYTHON_SLEEP", "2s")
		defer os.Unsetenv("FAKE_PYTHON_SLEEP") //nolint:errcheck

		err := eval.Run(context.Background(), dir, script, model, dataset, results, "", 50*time.Millisecond)
		if err == nil || !strings.Contains(err.Error(), "timed out") {
			t.Fatalf("Run() with a 50ms cap against a 2s sleep = %v, want a timeout error", err)
		}
	})

	t.Run("InstallDeps surfaces failing scanner output as DepsError, and succeeds on exit 0", func(t *testing.T) {
		scanner := filepath.Join(dir, "dep_scanner.py")
		if err := os.WriteFile(scanner, []byte("#stub\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		preprocessing := filepath.Join(dir, "user_preprocessing.py")
		if err := os.WriteFile(preprocessing, []byte("import nonexistent_lib\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		t.Setenv("FAKE_PYTHON_EXIT", "0")
		t.Setenv("FAKE_PYTHON_SLEEP", "")
		if err := eval.InstallDeps(context.Background(), dir, scanner, preprocessing, 0); err != nil {
			t.Fatalf("InstallDeps success case: %v", err)
		}

		t.Setenv("FAKE_PYTHON_EXIT", "1")
		t.Setenv("FAKE_PYTHON_STDERR", "ERROR: No matching distribution found for nonexistent-lib")
		defer os.Unsetenv("FAKE_PYTHON_STDERR") //nolint:errcheck
		err := eval.InstallDeps(context.Background(), dir, scanner, preprocessing, 0)
		var depsErr *eval.DepsError
		if !errors.As(err, &depsErr) {
			t.Fatalf("InstallDeps failure = %v (%T), want *eval.DepsError", err, err)
		}
		if !strings.Contains(depsErr.Output, "No matching distribution") {
			t.Errorf("DepsError.Output = %q, want it to carry the scanner's stderr tail", depsErr.Output)
		}
	})
}

// buildFakePython compiles a tiny stand-in "python3" binary the eval package
// invokes via exec.LookPath, so no real Python or uv is required to exercise
// the frozen CLI/exit-code contract. It supports the handful of env-var
// knobs the subtests above set: FAKE_PYTHON_EXIT, FAKE_PYTHON_SLEEP,
// FAKE_PYTHON_STDERR, FAKE_PYTHON_ARGV_FILE.
func buildFakePython(t *testing.T) string {
	t.Helper()
	srcDir := t.TempDir()
	src := filepath.Join(srcDir, "fakepython.go")
	source := `package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	if f := os.Getenv("FAKE_PYTHON_ARGV_FILE"); f != "" {
		os.WriteFile(f, []byte(strings.Join(os.Args[1:], " ")), 0o644)
	}
	if s := os.Getenv("FAKE_PYTHON_SLEEP"); s != "" {
		d, err := time.ParseDuration(s)
		if err == nil {
			time.Sleep(d)
		}
	}
	if msg := os.Getenv("FAKE_PYTHON_STDERR"); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
	}
	code := 0
	if c := os.Getenv("FAKE_PYTHON_EXIT"); c != "" {
		code, _ = strconv.Atoi(c)
	}
	os.Exit(code)
}
`
	if err := os.WriteFile(src, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	binDir := t.TempDir()
	bin := filepath.Join(binDir, "python3")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", bin, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building fake python3 stand-in: %v\n%s", err, out)
	}
	return bin
}

func prependPATH(t *testing.T, dir string) func() {
	t.Helper()
	old := os.Getenv("PATH")
	os.Setenv("PATH", dir+string(os.PathListSeparator)+old) //nolint:errcheck
	return func() { os.Setenv("PATH", old) }                //nolint:errcheck
}

// ---------------------------------------------------------------------
// 4. leaderboard — error classification and best-effort submission.
// ---------------------------------------------------------------------

func TestLeaderboard_ClassificationAndSubmission(t *testing.T) {
	t.Run("classifies wrapped eval errors correctly through errors.As", func(t *testing.T) {
		wrapped := fmt.Errorf("pipeline: run eval: %w", &eval.ExitCodeError{Code: 10})
		info := leaderboard.Classify(wrapped)
		if info.Code != 1 || info.Type != "EvalUserCodeError" {
			t.Errorf("wrapped exit-10 classified as %+v", info)
		}

		wrapped11 := fmt.Errorf("pipeline: %w", &eval.ExitCodeError{Code: 11})
		if info := leaderboard.Classify(wrapped11); info.Type != "EvalEnvironmentError" {
			t.Errorf("wrapped exit-11 classified as %+v", info)
		}

		notExist := fmt.Errorf("pipeline: read results.json: %w", fs.ErrNotExist)
		if info := leaderboard.Classify(notExist); info.Type != "FileNotFoundError" || info.Code != 3 {
			t.Errorf("wrapped fs.ErrNotExist classified as %+v", info)
		}
	})

	// Integrated from the leaderboard_test.go script
	t.Run("Classify: env-keyword scan and DepsError unwrap", func(t *testing.T) {
		cases := []struct {
			name     string
			err      error
			wantCode int
			wantType string
		}{
			{"unwrapped exit 10", &eval.ExitCodeError{Code: 10}, 1, "EvalUserCodeError"},
			{"unwrapped exit 11", &eval.ExitCodeError{Code: 11}, 3, "EvalEnvironmentError"},
			{"unwrapped other exit", &eval.ExitCodeError{Code: 7}, 2, "EvalScriptError"},
			{"message contains an env keyword", fmt.Errorf("connection to Secret Manager timed out"), 3, "EnvironmentError"},
			{"DepsError: user's bad import", &eval.DepsError{Output: "no such package: nonexistent-lib"}, 1, "PreprocessingDepsError"},
			{"DepsError: network failure during install", &eval.DepsError{Output: "connection reset while downloading"}, 3, "EnvironmentError"},
			{"unmatched message falls back to PipelineError", fmt.Errorf("something odd happened"), 2, "PipelineError"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				info := leaderboard.Classify(tc.err)
				if info.Code != tc.wantCode || info.Type != tc.wantType {
					t.Fatalf("Classify(%v) = (%d,%s), want (%d,%s)", tc.err, info.Code, info.Type, tc.wantCode, tc.wantType)
				}
			})
		}
	})

	// Integrated from the leaderboard_test.go script (TestSanitizeMsg)
	t.Run("SanitizeMsg redacts filesystem paths before external reporting", func(t *testing.T) {
		got := leaderboard.SanitizeMsg(`open /app/cvm_workflow/secure_jobs/j1/artifacts/model.onnx: no such file`)
		want := `open <path> no such file`
		if got != want {
			t.Fatalf("SanitizeMsg = %q, want %q", got, want)
		}
	})

	t.Run("MapMetrics per leaderboard vertical", func(t *testing.T) {
		oral := leaderboard.MapMetrics(2, map[string]any{
			"sensitivity": 0.92, "specificity": 0.88, "accuracy": 0.9,
			"ppv": 0.7, "npv": 0.95, "f2": 0.91, "junk_field": 1.0,
		})
		if oral["f2_score"] != 0.91 {
			t.Errorf("oral f2_score = %v, want 0.91", oral["f2_score"])
		}
		if _, ok := oral["junk_field"]; ok {
			t.Error("oral cancer mapping must not leak unknown fields")
		}

		unknownVertical := leaderboard.MapMetrics(3, map[string]any{"anything": 1.0, "gone": nil})
		if unknownVertical["anything"] != 1.0 {
			t.Error("unmapped dataset ids fall back to passthrough")
		}
		if _, ok := unknownVertical["gone"]; ok {
			t.Error("nil-valued fields must be omitted from the submitted metrics")
		}
	})

	// Integrated from the leaderboard_test.go script (TestMapMetricsBreastCancerDerivesWeightedF2)
	t.Run("MapMetrics breast-cancer vertical derives the support-weighted F2 golden value", func(t *testing.T) {
		in := map[string]any{
			"accuracy": 0.9,
			"per_class": map[string]any{
				// support 10, f2(precision=1, recall=1) = 1
				"a": map[string]any{"TP": 5.0, "FN": 5.0, "precision": 1.0, "recall": 1.0},
				// support 10, f2(precision=0.5, recall=0.5) = 5*0.25/(2+0.5) = 0.5
				"b": map[string]any{"TP": 2.0, "FN": 8.0, "precision": 0.5, "recall": 0.5},
			},
		}
		out := leaderboard.MapMetrics(1, in)
		wf2, ok := out["weighted_f2"].(float64)
		if !ok {
			t.Fatalf("weighted_f2 missing or wrong type: %v", out)
		}
		// Equal-support classes (10 and 10) at f2=1 and f2=0.5 average to 0.75.
		if diff := wf2 - 0.75; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("weighted_f2 = %v, want 0.75", wf2)
		}
		if _, ok := out["macro_f2"]; ok {
			t.Error("a nil macro_f2 should be omitted from the mapped metrics, not sent as null")
		}
	})

	// Integrated from the leaderboard_test.go script (TestUUID5MatchesPython)
	t.Run("golden: UUID5 matches the Python reference implementation", func(t *testing.T) {
		got := leaderboard.UUID5("tanuh:job-test-123")
		const golden = "1640b625-aa06-5f8c-a77a-befd07cc4020"
		if got != golden {
			t.Fatalf("UUID5 diverged from the Python golden:\n got  %s\n want %s", got, golden)
		}
	})

	t.Run("Submit is a no-op without a Keycloak token", func(t *testing.T) {
		var called int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&called, 1)
			w.WriteHeader(http.StatusCreated)
		}))
		defer srv.Close()

		leaderboard.Submit(context.Background(), srv.URL, leaderboard.Submission{
			JobID: "job-no-token", DatasetID: 1, Succeeded: true,
			Results: map[string]any{"metrics": map[string]any{}},
		})
		if atomic.LoadInt32(&called) != 0 {
			t.Error("Submit must skip the HTTP call entirely when no Keycloak token was forwarded")
		}
	})

	t.Run("Submit skips unmapped dataset ids", func(t *testing.T) {
		var called int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&called, 1)
		}))
		defer srv.Close()

		leaderboard.Submit(context.Background(), srv.URL, leaderboard.Submission{
			JobID: "job-x", DatasetID: 999, Succeeded: true, KeycloakToken: "t",
			Results: map[string]any{},
		})
		if atomic.LoadInt32(&called) != 0 {
			t.Error("Submit must skip datasets with no leaderboard vertical mapping")
		}
	})

	t.Run("Submit posts a success payload with attestation and Bearer auth", func(t *testing.T) {
		var gotAuth string
		var gotBody map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAuth = r.Header.Get("Authorization")
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &gotBody) //nolint:errcheck
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		leaderboard.Submit(context.Background(), srv.URL, leaderboard.Submission{
			JobID:     "job-happy",
			DatasetID: 2,
			Claims:    attest.Claims{HWModel: "GCP_AMD_SEV", SwName: "CONFIDENTIAL_SPACE", Secboot: true},
			Succeeded: true,
			Results: map[string]any{
				"num_samples":     100.0,
				"elapsed_seconds": 12.5,
				"metrics":         map[string]any{"sensitivity": 0.9},
			},
			KeycloakToken: "kc-token-abc",
		})

		if gotAuth != "Bearer kc-token-abc" {
			t.Errorf("Authorization header = %q", gotAuth)
		}
		if gotBody["status"] != "succeeded" {
			t.Errorf("status = %v, want succeeded", gotBody["status"])
		}
		if gotBody["dataset_id"] != "oral_cancer" {
			t.Errorf("dataset_id slug = %v, want oral_cancer", gotBody["dataset_id"])
		}
		attestation, _ := gotBody["attestation"].(map[string]any)
		if attestation["hwmodel"] != "GCP_AMD_SEV" || attestation["secboot"] != true {
			t.Errorf("attestation claims not forwarded correctly: %+v", attestation)
		}
	})

	t.Run("Submit posts a failure payload and never returns an error itself", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError) // even a broken leaderboard must not matter upstream
		}))
		defer srv.Close()

		errInfo := leaderboard.Classify(&eval.ExitCodeError{Code: 11})
		// Submit has no return value: reaching this line without panicking,
		// against a leaderboard that 500s, is the assertion.
		leaderboard.Submit(context.Background(), srv.URL, leaderboard.Submission{
			JobID: "job-sad", DatasetID: 1, Succeeded: false,
			Error: &errInfo, KeycloakToken: "kc-token",
		})
	})
}

// ---------------------------------------------------------------------
// 5. attest — CS claim decoding and the off-TEE degraded fallback.
// ---------------------------------------------------------------------

func TestAttest_ClaimsDecoding(t *testing.T) {
	t.Run("decodes a well-formed CS-shaped JWT payload", func(t *testing.T) {
		token := fakeJWT(t, map[string]any{
			"hwmodel": "GCP_AMD_SEV", "swname": "CONFIDENTIAL_SPACE",
			"iss": "https://confidentialcomputing.googleapis.com",
			"sub": "instance-123", "secboot": true, "iat": 1700000000.0,
			"submods": map[string]any{
				"container": map[string]any{
					"image_digest":    "sha256:abcd1234",
					"image_reference": "us-central1-docker.pkg.dev/p/r/i:latest",
				},
			},
		})
		claims := attest.DecodeJWTPayload(token)
		if claims == nil {
			t.Fatal("DecodeJWTPayload returned nil for a well-formed token")
		}
		if claims["hwmodel"] != "GCP_AMD_SEV" {
			t.Errorf("hwmodel = %v", claims["hwmodel"])
		}
	})

	t.Run("returns nil rather than panicking on malformed tokens", func(t *testing.T) {
		for name, tok := range map[string]string{
			"wrong segment count": "onlyonepart",
			"invalid base64":      "aaa.not-valid-base64!!!.ccc",
			"invalid JSON payload": "aaa." +
				base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString([]byte("not json")) + ".ccc",
			"empty string": "",
		} {
			t.Run(name, func(t *testing.T) {
				if got := attest.DecodeJWTPayload(tok); got != nil {
					t.Errorf("DecodeJWTPayload(%q) = %v, want nil", tok, got)
				}
			})
		}
	})

	t.Run("FetchClaims degrades to zero-value Claims off-TEE without erroring", func(t *testing.T) {
		// Neither the CS launcher's unix socket nor the GCP metadata server
		// exist in this test environment, so
		// this exercises the exact fallback chain FetchClaims documents:
		// launcher socket -> metadata identity token -> zero Claims{}.
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		claims := attest.FetchClaims(ctx, "https://tanuh-processing-tee")
		if claims != (attest.Claims{}) {
			t.Errorf("FetchClaims off-TEE = %+v, want the zero value", claims)
		}
	})
}

func fakeJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	header := base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	body, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	payload := base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString(body)
	return header + "." + payload + ".unverified-signature"
}

// ---------------------------------------------------------------------
// 6. policy — deterministic hashing and fail-closed boot attestation.
// ---------------------------------------------------------------------

func TestPolicy_HashingAndStartupAttestation(t *testing.T) {
	t.Run("canonical JSON sorts keys, compacts, and does not HTML-escape", func(t *testing.T) {
		v := map[string]any{
			"z": "a<b>c&d",
			"a": []any{map[string]any{"y": 2.0, "x": 1.0}},
		}
		got, err := policy.CanonicalJSON(v)
		if err != nil {
			t.Fatal(err)
		}
		want := `{"a":[{"x":1,"y":2}],"z":"a<b>c&d"}`
		if got != want {
			t.Fatalf("got %q want %q", got, want)
		}
	})

	t.Run("Hash is deterministic and input-sensitive", func(t *testing.T) {
		p1 := map[string]any{"rules": []any{"a"}}
		p2 := map[string]any{"rules": []any{"b"}}
		h1a, _ := policy.Hash(p1)
		h1b, _ := policy.Hash(p1)
		h2, _ := policy.Hash(p2)
		if h1a != h1b {
			t.Error("Hash must be deterministic for identical input")
		}
		if h1a == h2 {
			t.Error("Hash must differ for different policy content")
		}
	})

	t.Run("Load surfaces missing-file and invalid-JSON errors", func(t *testing.T) {
		if _, err := policy.Load(filepath.Join(t.TempDir(), "missing.json")); err == nil {
			t.Error("Load of a nonexistent file should error")
		}
		bad := filepath.Join(t.TempDir(), "bad.json")
		if err := os.WriteFile(bad, []byte("{not json"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := policy.Load(bad); err == nil {
			t.Error("Load of invalid JSON should error")
		}
	})

	t.Run("StartupAttestation fails closed off-TEE", func(t *testing.T) {
		// Mirrors main.go's boot contract: if the policy hash cannot be
		// bound into an attestation token, the server must never start
		// listening. Off-TEE (no launcher socket) that's exactly what
		// should happen here.
		dir := t.TempDir()
		policyPath := filepath.Join(dir, "network_policy.json")
		if err := os.WriteFile(policyPath, []byte(`{"rules":[]}`), 0o644); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := policy.StartupAttestation(ctx, policyPath, "ratls-buffer-tee"); err == nil {
			t.Error("StartupAttestation should fail off-TEE (no CS launcher socket available)")
		}
	})

	// Integrated from the policy_test.go script (TestHashMatchesPython)
	t.Run("shipped network_policy.json hashes to the Python golden", func(t *testing.T) {
		p, err := policy.Load(filepath.Join("..", "policy", "network_policy.json"))
		if err != nil {
			t.Fatalf("load shipped policy: %v", err)
		}
		hash, err := policy.Hash(p)
		if err != nil {
			t.Fatal(err)
		}
		const golden = "e0973010fab9192b4275d84e1403a573cdf84e0a70d47c8bfe95af525de4a10e"
		if hash != golden {
			t.Fatalf("shipped policy hash diverged from the Python golden:\n got  %s\n want %s\n(if you intentionally edited policy/network_policy.json, recompute this golden with the command above)", hash, golden)
		}
	})
}

// ---------------------------------------------------------------------
// 7. ratls — the EKM channel-binding contract between two independent
//    TLS 1.3 stacks, plus the server's fail-closed behaviour off-TEE.
// ---------------------------------------------------------------------

func TestRATLS_EKMChannelBindingAndConnectHandler(t *testing.T) {
	t.Run("ExtractEKM rejects an incomplete handshake state", func(t *testing.T) {
		if _, err := ratls.ExtractEKM(nil); err == nil {
			t.Error("ExtractEKM(nil) should error")
		}
	})

	t.Run("both sides of one TLS 1.3 session derive identical EKM bytes", func(t *testing.T) {
		rs, err := ratls.NewServer("test-audience")
		if err != nil {
			t.Fatalf("NewServer: %v", err)
		}
		serverState, clientState := handshakeOverPipe(t, rs.TLSConfig())

		serverEKM, err := ratls.ExtractEKM(serverState)
		if err != nil {
			t.Fatalf("server ExtractEKM: %v", err)
		}
		clientEKM, err := ratls.ExtractEKM(clientState)
		if err != nil {
			t.Fatalf("client ExtractEKM: %v", err)
		}
		if len(serverEKM) != 32 {
			t.Fatalf("EKM length = %d, want 32", len(serverEKM))
		}
		if !bytes.Equal(serverEKM, clientEKM) {
			t.Fatal("server and client must derive the same EKM from the same session — this is the entire RA-TLS channel-binding guarantee")
		}

		nonce := ratls.EKMNonce(serverEKM)
		if len(nonce) != 43 {
			t.Errorf("EKMNonce length = %d, want 43 (base64url, no padding, of a 32-byte hash)", len(nonce))
		}
		if strings.ContainsAny(nonce, "+/=") {
			t.Errorf("EKMNonce %q contains non-URL-safe or padding characters", nonce)
		}
		if got := ratls.EKMNonce(serverEKM); got != nonce {
			t.Error("EKMNonce must be deterministic for the same EKM input")
		}
	})

	t.Run("/ratls/connect fails closed (500) off-TEE, after a real TLS 1.3 handshake", func(t *testing.T) {
		rs, err := ratls.NewServer("test-audience")
		if err != nil {
			t.Fatalf("NewServer: %v", err)
		}
		mux := http.NewServeMux()
		mux.Handle(ratls.ConnectPath, rs.ConnectHandler())

		srv := httptest.NewUnstartedServer(mux)
		srv.TLS = rs.TLSConfig()
		srv.StartTLS()
		defer srv.Close()

		// The server presents ratls.Server's own self-signed cert (its CA
		// is deliberately irrelevant — see NewServer's doc comment: clients
		// authenticate the session via the attestation token's EKM channel
		// binding, not the certificate chain), not httptest's default one,
		// so httptest's own srv.Client() would fail to verify it.
		client := &http.Client{Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS13}, //nolint:gosec
		}}
		resp, err := client.Get(srv.URL + ratls.ConnectPath)
		if err != nil {
			t.Fatalf("GET %s: %v", ratls.ConnectPath, err)
		}
		defer resp.Body.Close()

		// The handshake must have completed (we got an HTTP response at
		// all) and the handler must have reached the point of asking the
		// (absent) CS launcher for a token, and failed closed rather than
		// return a forged/empty attestation bundle.
		if resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("status = %d, want 500 (attestation error) since no CS launcher socket is present", resp.StatusCode)
		}
	})
}

// handshakeOverPipe drives a real TLS 1.3 client/server handshake over an
// in-memory net.Pipe using the server's actual RA-TLS TLSConfig, and returns
// both sides' post-handshake ConnectionState.
func handshakeOverPipe(t *testing.T, serverCfg *tls.Config) (server, client *tls.ConnectionState) {
	t.Helper()
	c1, c2 := net.Pipe()

	clientCfg := &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS13} //nolint:gosec
	serverConn := tls.Server(c1, serverCfg)
	clientConn := tls.Client(c2, clientCfg)

	errCh := make(chan error, 2)
	go func() { errCh <- serverConn.Handshake() }()
	go func() { errCh <- clientConn.Handshake() }()

	for i := 0; i < 2; i++ {
		if err := <-errCh; err != nil {
			t.Fatalf("TLS handshake: %v", err)
		}
	}
	sState := serverConn.ConnectionState()
	cState := clientConn.ConnectionState()
	// tls.Conn.Close() sends a close_notify alert and blocks on the
	// underlying Write; net.Pipe is synchronous and unbuffered, and
	// neither side is reading post-handshake, so a graceful TLS close
	// here would stall. Close the raw pipe halves instead — the
	// ConnectionState values above are already captured.
	t.Cleanup(func() {
		c1.Close() //nolint:errcheck
		c2.Close() //nolint:errcheck
	})
	return &sState, &cState
}

// ---------------------------------------------------------------------
// 8. server + pipeline — the full HTTP-in-to-terminal-state workflow.
//    This is the "does the whole project fit together" scenario: a
//    dispatch payload lands on /api/load-model exactly the way the
//    Buffer TEE sends it, runs the real (unmocked) pipeline, and — since
//    GCP is unreachable in a test environment — is driven all the way to
//    its real fail-closed terminal state, persisted to disk exactly as
//    cmd/processing-tee/main.go and the README's Debug section describe.
// ---------------------------------------------------------------------

func TestServerAndPipeline_FullJobWorkflow(t *testing.T) {
	base := t.TempDir()
	cfg := config.Config{
		BaseDir:           base,
		ListenAddr:        ":0",
		RATLSAudience:     "test-audience",
		ProjectID:         "test-project",
		DatasetsBucket:    "test-datasets-bucket",
		EvalScriptsBucket: "test-eval-scripts-bucket",
		Datasets: map[int]config.Dataset{
			1: {
				JSONObject:       "breast-cancer/dataset.json.enc",
				ImagesObject:     "breast-cancer/images.zip.enc",
				SecretID:         "tanuh-ds1-key",
				EvalScriptObject: "evaluate_model_breastcancer.py",
				ImagesExtractDir: filepath.Join(base, "cvm_workflow/data/breast-cancer"),
			},
		},
		StopProject:      "test-project",
		StopZone:         "us-central1-a",
		StopInstance:     "test-instance",
		IdleTimeout:      time.Hour, // keep the idle deallocator out of this test
		DeallocAfterJob:  true,
		EvalTimeout:      0,
		DepsTimeout:      0,
		LeaderboardURL:   "https://example.invalid/leaderboard/submit-solution",
		AttestAudience:   "https://tanuh-processing-tee",
		CallbackAudience: "tanuh-buffer-callback",
		CallbackTimeout:  2 * time.Second,
	}

	mgr, err := pipeline.NewManager(cfg)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	rs, err := ratls.NewServer(cfg.RATLSAudience)
	if err != nil {
		t.Fatalf("ratls.NewServer: %v", err)
	}
	incomingDir := filepath.Join(cfg.WorkflowDir(), "incoming")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	handler := server.New(rs, mgr, incomingDir, ctx)
	httpSrv := httptest.NewServer(handler)
	defer httpSrv.Close()

	t.Run("/healthz reports OK", func(t *testing.T) {
		resp, err := http.Get(httpSrv.URL + "/healthz")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("healthz status = %d, want 200", resp.StatusCode)
		}
	})

	t.Run("load-model rejects invalid JSON", func(t *testing.T) {
		resp, err := http.Post(httpSrv.URL+"/api/load-model", "application/json", strings.NewReader("{not json"))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", resp.StatusCode)
		}
	})

	t.Run("load-model rejects a missing job_id", func(t *testing.T) {
		resp, err := http.Post(httpSrv.URL+"/api/load-model", "application/json", strings.NewReader(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", resp.StatusCode)
		}
	})

	// The centrepiece scenario: a structurally valid dispatch payload (the
	// SHA-256 commitments are real and correct, exactly like the browser's
	// commitment in materialize()'s doc comment) for a configured dataset,
	// but with GCP entirely unreachable. This drives the job through
	// materialize -> runSteps -> datasetKey (fails) -> Classify -> attest
	// fallback -> leaderboard (skipped, no keycloak token) -> notifyBuffer
	// (skipped, no buffer_job_url) -> RequestDeallocation (also fails,
	// same unreachable network) -> terminal persisted state.
	t.Run("a full dispatch payload is accepted, then reaches a fail-closed terminal state", func(t *testing.T) {
		jobID := "job-e2e-workflow-1"
		payload := buildSignedPayload(t, jobID, 1)

		resp, err := http.Post(httpSrv.URL+"/api/load-model", "application/json", bytes.NewReader(payload))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusAccepted {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 202: %s", resp.StatusCode, body)
		}
		var accepted map[string]string
		if err := json.NewDecoder(resp.Body).Decode(&accepted); err != nil {
			t.Fatal(err)
		}
		if accepted["job_id"] != jobID || accepted["status"] != "accepted" {
			t.Errorf("accept body = %+v", accepted)
		}

		receiptPath := filepath.Join(incomingDir, jobID+"-secure-job.json")
		if _, err := os.Stat(receiptPath); err != nil {
			t.Errorf("incoming payload receipt not written: %v", err)
		}

		// A second dispatch while the first job holds the slot must 409.
		resp2, err := http.Post(httpSrv.URL+"/api/load-model", "application/json",
			strings.NewReader(`{"job_id":"job-e2e-workflow-2"}`))
		if err != nil {
			t.Fatal(err)
		}
		defer resp2.Body.Close()
		if resp2.StatusCode != http.StatusConflict {
			t.Errorf("concurrent dispatch status = %d, want 409", resp2.StatusCode)
		}

		state := waitForTerminalState(t, filepath.Join(cfg.WorkflowDir(), "runtime", "secure_runtime_state.json"), 25*time.Second)
		if state.LastJobID != jobID {
			t.Errorf("LastJobID = %q, want %q", state.LastJobID, jobID)
		}
		if state.Status != "error" && state.Status != "deallocation_failed" {
			t.Errorf("terminal Status = %q, want error or deallocation_failed (GCP is unreachable in this test)", state.Status)
		}
		if state.LastError == "" {
			t.Error("LastError should be populated once the dataset key fetch fails")
		}
		if state.CurrentJobID != "" {
			t.Error("CurrentJobID should be cleared once the job finishes")
		}
	})
}

// buildSignedPayload constructs a structurally valid /api/load-model
// dispatch body: small placeholder model/weights blobs with correct
// SHA-256 commitments, matching materialize()'s fail-closed verification.
func buildSignedPayload(t *testing.T, jobID string, datasetID int) []byte {
	t.Helper()
	model := []byte("fake-onnx-model-bytes")
	weights := []byte("fake-onnx-weights-bytes")
	modelSum := sha256.Sum256(model)
	weightsSum := sha256.Sum256(weights)

	payload := map[string]any{
		"job_id":               jobID,
		"dataset_id":           datasetID,
		"model_onnx_base64":    base64.StdEncoding.EncodeToString(model),
		"model_weights_base64": base64.StdEncoding.EncodeToString(weights),
		"model_sha256":         hex.EncodeToString(modelSum[:]),
		"weights_sha256":       hex.EncodeToString(weightsSum[:]),
		"submitted_by":         "integration-test@example.invalid",
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// waitForTerminalState polls the manager's persisted state file (the same
// file the README's Debug section documents) until its Status stops being
// one of the in-flight values, or the deadline expires.
func waitForTerminalState(t *testing.T, path string, timeout time.Duration) pipeline.State {
	t.Helper()
	deadline := time.Now().Add(timeout)
	inFlight := map[string]bool{
		"waiting_for_job": true, "job_received": true, "running": true,
	}
	var last pipeline.State
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil {
			var s pipeline.State
			if json.Unmarshal(raw, &s) == nil {
				last = s
				if !inFlight[s.Status] && s.Status != "" {
					return s
				}
			}
		}
		time.Sleep(150 * time.Millisecond)
	}
	t.Fatalf("state never reached a terminal status within %s; last observed: %+v", timeout, last)
	return last
}
