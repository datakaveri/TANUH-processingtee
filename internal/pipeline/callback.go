package pipeline

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
	"github.com/datakaveri/tanuh-processing-tee/internal/leaderboard"
)

// notifyBuffer POSTs the job's terminal status to the Buffer TEE's
// completion endpoint (buffer_job_url + "/complete", forwarded in the
// dispatch payload — served on the buffer's :8443 Go server).
//
// Authentication is this workload's own CS attestation token, requested
// with eat_nonce = hex(sha256(request body)): the buffer verifies the token
// against Google's CS JWKS, checks our image_digest against the digests it
// dispatches to, and recomputes the body hash — so only the attested
// Processing TEE can complete jobs, and the token cannot be replayed with a
// different payload. Best-effort: the buffer's bounded requeue-on-timeout
// remains the fallback when the callback cannot be delivered.
func (m *Manager) notifyBuffer(ctx context.Context, bufferJobURL, jobID, status string, errInfo *leaderboard.ErrorInfo) {
	if bufferJobURL == "" {
		log.Printf("pipeline: no buffer_job_url in payload for job %s; skipping completion callback", jobID)
		return
	}
	body := map[string]any{
		"job_id": jobID,
		"status": status, // "succeeded" | "failed"
	}
	if errInfo != nil {
		body["error"] = errInfo
	}
	payload, err := json.Marshal(body)
	if err != nil {
		log.Printf("pipeline: marshal completion callback for job %s: %v", jobID, err)
		return
	}

	// Bind the attestation token to this exact payload.
	bodySum := sha256.Sum256(payload)
	token, err := attest.TokenViaSocket(ctx, m.cfg.CallbackAudience, []string{hex.EncodeToString(bodySum[:])})
	if err != nil {
		log.Printf("pipeline: completion callback for job %s skipped — no attestation token: %v", jobID, err)
		return
	}

	url := bufferJobURL + "/complete"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		log.Printf("pipeline: build completion callback for job %s: %v", jobID, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{Timeout: m.cfg.CallbackTimeout}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("pipeline: completion callback for job %s failed (non-fatal): %v", jobID, err)
		return
	}
	defer resp.Body.Close()
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 200))
	log.Printf("pipeline: completion callback for job %s (%s) → %d %s", jobID, status, resp.StatusCode, string(snippet))
}
