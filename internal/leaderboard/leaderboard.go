// Package leaderboard builds and posts evaluation results to the external
// benchmark leaderboard (/leaderboard/submit-solution). Submission is
// best-effort: a failed submit never fails the pipeline. Authentication is
// the submitting user's Keycloak Bearer JWT, forwarded through the dispatch
// payload — the leaderboard requires its org_admin realm role; the CS
// attestation token is NOT accepted there. Attestation claims ride in the
// body so each entry carries its TEE provenance.
package leaderboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
)

// Submission carries everything needed for one leaderboard POST.
type Submission struct {
	JobID         string
	DatasetID     int
	Claims        attest.Claims
	Succeeded     bool
	Results       map[string]any // required when Succeeded
	Error         *ErrorInfo     // optional when !Succeeded
	KeycloakToken string
}

// Submit POSTs one evaluation result. Never returns an error — outcomes are
// logged, and the pipeline's success does not depend on it.
func Submit(ctx context.Context, submitURL string, s Submission) {
	slug, ok := datasetSlug[s.DatasetID]
	if !ok {
		log.Printf("leaderboard: no vertical for dataset_id=%d; skipping submit", s.DatasetID)
		return
	}
	// The leaderboard authenticates the caller's Keycloak Bearer JWT. If the
	// browser did not forward one, skip rather than send a token the
	// leaderboard will reject.
	if s.KeycloakToken == "" {
		log.Printf("leaderboard: no Keycloak token forwarded for job %s; skipping submit", s.JobID)
		return
	}

	attestation := map[string]any{
		"hwmodel":      s.Claims.HWModel,
		"swname":       s.Claims.SwName,
		"image_digest": s.Claims.ImageDigest,
		"secboot":      s.Claims.Secboot,
		"iss":          s.Claims.Iss,
	}

	body := map[string]any{
		"job_id":      UUID5("tanuh:" + s.JobID),
		"dataset_id":  slug,
		"attestation": attestation,
	}
	status := "failed"
	if s.Succeeded && s.Results != nil {
		status = "succeeded"
		metrics, _ := s.Results["metrics"].(map[string]any)
		body["num_samples"] = s.Results["num_samples"]
		body["elapsed_seconds"] = s.Results["elapsed_seconds"]
		body["model_sha256"] = s.Results["model_sha256"]
		providers := s.Results["onnx_runtime_providers"]
		if providers == nil {
			providers = []any{}
		}
		body["onnx_runtime_providers"] = providers
		body["metrics"] = MapMetrics(s.DatasetID, metrics)
	} else if s.Error != nil {
		body["error"] = s.Error
	}
	body["status"] = status

	payload, err := json.Marshal(body)
	if err != nil {
		log.Printf("leaderboard: marshal submit body for job %s: %v", s.JobID, err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, submitURL, bytes.NewReader(payload))
	if err != nil {
		log.Printf("leaderboard: build submit request for job %s: %v", s.JobID, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.KeycloakToken)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("leaderboard: submit failed for job %s (non-fatal): %v", s.JobID, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated {
		log.Printf("leaderboard: %s submitted for job %s (%s) → %d", status, s.JobID, slug, resp.StatusCode)
	} else {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		log.Printf("leaderboard: submit for job %s returned %d: %s", s.JobID, resp.StatusCode, string(snippet))
	}
}

// String implements fmt.Stringer for logging.
func (e ErrorInfo) String() string {
	return fmt.Sprintf("code=%d type=%s msg=%s", e.Code, e.Type, e.Message)
}
