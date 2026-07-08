package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
	"github.com/datakaveri/tanuh-processing-tee/internal/leaderboard"
)

const (
	modelFileName   = "model.onnx"
	weightsFileName = "model.onnx.data"
)

// materialized holds everything RunJob needs after payload validation.
type materialized struct {
	jobID             string
	datasetID         int
	jobDir            string
	runtimeDir        string
	modelPath         string
	weightsPath       string
	preprocessingPath string
	submittedBy       string
	keycloakToken     string
	bufferJobURL      string
}

// RunJob executes the full secure-job pipeline for an accepted payload and
// always releases the job slot. Meant to run on its own goroutine.
func (m *Manager) RunJob(ctx context.Context, payload map[string]any) {
	jobID := stringField(payload, "job_id")
	defer m.releaseJobSlot()

	job, err := m.materialize(payload)
	if err == nil {
		err = m.runSteps(ctx, job)
	}
	if err == nil {
		return
	}

	// Failure path: log the full error inside the TEE (serial log only),
	// report a sanitised classification externally, notify the buffer, and
	// deallocate so a failed job never leaves the VM running.
	log.Printf("pipeline: secure job %s failed: %v", jobID, err)
	m.setState(func(s *State) {
		s.Status = "error"
		s.CurrentJobID = ""
		s.LastJobID = jobID
		s.LastError = err.Error()
	})

	errInfo := leaderboard.Classify(err)
	claims := attest.FetchClaims(ctx, m.cfg.AttestAudience)
	leaderboard.Submit(ctx, m.cfg.LeaderboardURL, leaderboard.Submission{
		JobID:         jobID,
		DatasetID:     datasetIDField(payload),
		Claims:        claims,
		Succeeded:     false,
		Error:         &errInfo,
		KeycloakToken: stringField(payload, "keycloak_token"),
	})
	m.notifyBuffer(ctx, stringField(payload, "buffer_job_url"), jobID, "failed", &errInfo)

	if m.cfg.DeallocAfterJob {
		m.RequestDeallocation(ctx, fmt.Sprintf("job %s failed", jobID))
	}
}

// runSteps is the happy path: dataset → images → eval script → eval →
// results → leaderboard → buffer callback → dealloc.
func (m *Manager) runSteps(ctx context.Context, job *materialized) error {
	m.setState(func(s *State) {
		s.Status = "running"
		s.CurrentJobID = job.jobID
		s.LastJobID = job.jobID
		s.LastError = ""
	})
	log.Printf("pipeline: secure job %s: starting pipeline for dataset_id=%d", job.jobID, job.datasetID)

	dataset, err := m.cfg.Dataset(job.datasetID)
	if err != nil {
		return err
	}

	// Step 1: fetch the AES-256 dataset key and the encrypted dataset JSON,
	// decrypt in memory (it is small).
	key, err := m.datasetKey(ctx, dataset.SecretID)
	if err != nil {
		return err
	}
	datasetPath, err := m.decryptDatasetJSON(ctx, key, dataset, job)
	if err != nil {
		return err
	}
	log.Printf("pipeline: secure job %s: dataset decrypted to %s", job.jobID, datasetPath)

	// Step 2: fetch the encrypted image zip, decrypt (chunked stream), extract.
	if err := m.fetchAndExtractImages(ctx, key, dataset, job.datasetID); err != nil {
		return err
	}

	// Step 3: pull the dataset's evaluation script.
	scriptPath := filepath.Join(m.workflowDir(), fmt.Sprintf("evaluation_script_%d.py", job.datasetID))
	log.Printf("pipeline: secure job %s: fetching eval script gs://%s/%s",
		job.jobID, m.cfg.EvalScriptsBucket, dataset.EvalScriptObject)
	if err := gcp.DownloadObject(ctx, m.cfg.EvalScriptsBucket, dataset.EvalScriptObject, scriptPath); err != nil {
		return err
	}

	// Step 4: when the user sent a preprocessing script, install any
	// third-party imports the image doesn't ship (dep_scanner.py + uv)
	// before the eval subprocess launches.
	if job.preprocessingPath != "" {
		if err := eval.InstallDeps(ctx, m.cfg.BaseDir,
			filepath.Join(m.cfg.BaseDir, "dep_scanner.py"),
			job.preprocessingPath, m.cfg.DepsTimeout); err != nil {
			return err
		}
	}

	// Step 5: run the evaluation subprocess.
	resultsPath := filepath.Join(job.runtimeDir, "results.json")
	if err := eval.Run(ctx, m.cfg.BaseDir, scriptPath, job.modelPath, datasetPath,
		resultsPath, job.preprocessingPath, m.cfg.EvalTimeout); err != nil {
		return err
	}

	results, err := m.enrichResults(resultsPath, job)
	if err != nil {
		return err
	}
	log.Printf("pipeline: secure job %s: results saved to %s", job.jobID, resultsPath)

	// Step 5: report — attestation claims ride in the leaderboard body.
	log.Printf("pipeline: secure job %s: fetching attestation claims for leaderboard", job.jobID)
	claims := attest.FetchClaims(ctx, m.cfg.AttestAudience)
	leaderboard.Submit(ctx, m.cfg.LeaderboardURL, leaderboard.Submission{
		JobID:         job.jobID,
		DatasetID:     job.datasetID,
		Claims:        claims,
		Succeeded:     true,
		Results:       results,
		KeycloakToken: job.keycloakToken,
	})
	m.notifyBuffer(ctx, job.bufferJobURL, job.jobID, "succeeded", nil)

	m.setState(func(s *State) {
		s.Status = "complete"
		s.CurrentJobID = ""
		s.LastJobID = job.jobID
		s.LastError = ""
	})
	if m.cfg.DeallocAfterJob {
		m.RequestDeallocation(ctx, fmt.Sprintf("job %s finished successfully", job.jobID))
	}
	return nil
}

// materialize validates the payload and writes model/weights/preprocessing
// to the job's artifacts dir. Integrity is SHA-256 committed by the browser
// at submit time and re-verified here; confidentiality in transit is the
// RA-TLS channel (there is no app-layer payload encryption — the former
// keystore path was never wired and has been removed).
func (m *Manager) materialize(payload map[string]any) (*materialized, error) {
	for _, f := range []string{"job_id", "dataset_id", "model_onnx_base64",
		"model_weights_base64", "model_sha256", "weights_sha256"} {
		if _, ok := payload[f]; !ok {
			return nil, fmt.Errorf("pipeline: missing field in secure job payload: %s", f)
		}
	}

	job := &materialized{
		jobID:         stringField(payload, "job_id"),
		datasetID:     datasetIDField(payload),
		submittedBy:   stringField(payload, "submitted_by"),
		keycloakToken: stringField(payload, "keycloak_token"),
		bufferJobURL:  stringField(payload, "buffer_job_url"),
	}
	if job.jobID == "" {
		return nil, fmt.Errorf("pipeline: empty job_id")
	}
	if job.datasetID == 0 {
		return nil, fmt.Errorf("pipeline: invalid dataset_id")
	}

	job.jobDir = m.jobDir(job.jobID)
	artifactsDir := filepath.Join(job.jobDir, "artifacts")
	job.runtimeDir = filepath.Join(job.jobDir, "runtime")
	for _, d := range []string{artifactsDir, job.runtimeDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}

	modelBytes, err := decodeAndVerify(payload, "model_onnx_base64", "model_sha256")
	if err != nil {
		return nil, err
	}
	weightsBytes, err := decodeAndVerify(payload, "model_weights_base64", "weights_sha256")
	if err != nil {
		return nil, err
	}

	job.modelPath = filepath.Join(artifactsDir, fileNameField(payload, "model_file", modelFileName))
	job.weightsPath = filepath.Join(artifactsDir, fileNameField(payload, "weights_file", weightsFileName))
	if err := os.WriteFile(job.modelPath, modelBytes, 0o644); err != nil {
		return nil, err
	}
	if err := os.WriteFile(job.weightsPath, weightsBytes, 0o644); err != nil {
		return nil, err
	}

	if pp := stringField(payload, "preprocessing_script_base64"); pp != "" {
		ppBytes, err := decodeAndVerify(payload, "preprocessing_script_base64", "preprocessing_sha256")
		if err != nil {
			return nil, err
		}
		job.preprocessingPath = filepath.Join(artifactsDir, "preprocessing.py")
		if err := os.WriteFile(job.preprocessingPath, ppBytes, 0o644); err != nil {
			return nil, err
		}
		log.Printf("pipeline: preprocessing.py written (%d bytes)", len(ppBytes))
	}

	raw, err := json.MarshalIndent(payload, "", "  ")
	if err == nil {
		if err := os.WriteFile(filepath.Join(job.jobDir, "incoming_payload.json"), raw, 0o644); err != nil {
			log.Printf("pipeline: write incoming_payload.json: %v", err)
		}
	}
	log.Printf("pipeline: secure job payload materialized under %s", job.jobDir)
	return job, nil
}

// datasetKey fetches and decodes the hex AES-256 key from Secret Manager.
func (m *Manager) datasetKey(ctx context.Context, secretID string) ([]byte, error) {
	log.Printf("pipeline: fetching AES-256 key from Secret Manager: %s", secretID)
	keyHex, err := gcp.AccessSecret(ctx, m.cfg.ProjectID, secretID)
	if err != nil {
		return nil, err
	}
	key, err := hex.DecodeString(strings.TrimSpace(string(keyHex)))
	if err != nil {
		return nil, fmt.Errorf("pipeline: dataset key is not valid hex: %w", err)
	}
	if len(key) != 32 {
		return nil, fmt.Errorf("pipeline: dataset key is %d bytes, want 32", len(key))
	}
	return key, nil
}

func stringField(payload map[string]any, key string) string {
	s, _ := payload[key].(string)
	return s
}

// fileNameField returns a payload-provided file name, rejecting anything
// with path separators (defence against path traversal from the payload).
func fileNameField(payload map[string]any, key, def string) string {
	v := stringField(payload, key)
	if v == "" || v != filepath.Base(v) {
		return def
	}
	return v
}

func datasetIDField(payload map[string]any) int {
	switch v := payload["dataset_id"].(type) {
	case float64:
		return int(v)
	case string:
		n, _ := strconv.Atoi(v)
		return n
	case int:
		return v
	}
	return 0
}

// decodeAndVerify base64-decodes payload[b64Key] and enforces the SHA-256
// commitment in payload[shaKey] (fail closed on mismatch; an absent
// commitment is only tolerated for optional artifacts whose sha key is empty).
func decodeAndVerify(payload map[string]any, b64Key, shaKey string) ([]byte, error) {
	data, err := base64.StdEncoding.DecodeString(stringField(payload, b64Key))
	if err != nil {
		return nil, fmt.Errorf("pipeline: %s is not valid base64: %w", b64Key, err)
	}
	expected := strings.ToLower(strings.TrimSpace(stringField(payload, shaKey)))
	if expected != "" {
		sum := sha256.Sum256(data)
		if hex.EncodeToString(sum[:]) != expected {
			return nil, fmt.Errorf("pipeline: %s SHA256 mismatch in secure payload", b64Key)
		}
	}
	return data, nil
}

// enrichResults loads results.json, stamps identity fields, writes it back,
// and returns the parsed map for the leaderboard submission.
func (m *Manager) enrichResults(resultsPath string, job *materialized) (map[string]any, error) {
	raw, err := os.ReadFile(resultsPath)
	if err != nil {
		return nil, fmt.Errorf("pipeline: read results.json: %w", err)
	}
	var results map[string]any
	if err := json.Unmarshal(raw, &results); err != nil {
		return nil, fmt.Errorf("pipeline: parse results.json: %w", err)
	}

	modelSHA, err := sha256File(job.modelPath)
	if err != nil {
		return nil, err
	}
	weightsSHA, err := sha256File(job.weightsPath)
	if err != nil {
		return nil, err
	}
	results["job_id"] = job.jobID
	results["model_sha256"] = modelSHA
	results["weights_sha256"] = weightsSHA
	results["dataset_id"] = job.datasetID

	out, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(resultsPath, out, 0o644); err != nil {
		return nil, err
	}
	return results, nil
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
