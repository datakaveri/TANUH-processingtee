// Package server wires the Processing TEE's external HTTP surface: the
// RA-TLS connect endpoint, the job intake (/api/load-model, the path the
// Buffer TEE dispatch client POSTs to), and the health check the buffer
// polls while provisioning. Everything runs in-process — the former
// Flask :4000 hop and the :8081 loopback key API are gone.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/datakaveri/tanuh-processing-tee/internal/pipeline"
	"github.com/datakaveri/tanuh-processing-tee/internal/ratls"
)

// maxPayloadBytes bounds the dispatch payload (base64 model + weights).
const maxPayloadBytes = 2 << 30 // 2 GiB

// New builds the ServeMux for the external RA-TLS listener.
func New(rs *ratls.Server, mgr *pipeline.Manager, incomingDir string, jobCtx context.Context) http.Handler {
	mux := http.NewServeMux()
	mux.Handle(ratls.ConnectPath, rs.ConnectHandler())
	mux.HandleFunc("POST /api/load-model", handleLoadModel(mgr, incomingDir, jobCtx))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return mux
}

// handleLoadModel accepts a secure job payload and starts the pipeline on
// its own goroutine. 202 on acceptance, 409 while another job is running —
// the same contract the Flask manager exposed via the old Go proxy.
func handleLoadModel(mgr *pipeline.Manager, incomingDir string, jobCtx context.Context) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, maxPayloadBytes))
		if err != nil {
			jsonError(w, http.StatusBadRequest, "read payload failed")
			return
		}
		log.Printf("server: load-model request from %s (%d bytes)", r.RemoteAddr, len(body))

		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			jsonError(w, http.StatusBadRequest, "invalid JSON job payload")
			return
		}
		jobID, _ := payload["job_id"].(string)
		if jobID == "" {
			jsonError(w, http.StatusBadRequest, "job_id is required")
			return
		}

		ok, current := mgr.TryStartJob(jobID)
		if !ok {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			json.NewEncoder(w).Encode(map[string]string{ //nolint:errcheck
				"status":  "busy",
				"message": fmt.Sprintf("Already running job %s", current),
			})
			return
		}

		// Persist the raw payload as a receipt (same as the old incoming dir).
		if err := os.WriteFile(filepath.Join(incomingDir, jobID+"-secure-job.json"), body, 0o644); err != nil {
			log.Printf("server: write incoming payload receipt: %v", err)
		}

		// The job outlives this request — run it on the server's lifetime
		// context, not the request's.
		go mgr.RunJob(jobCtx, payload)

		log.Printf("server: secure job %s accepted from RA-TLS dispatch", jobID)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{ //nolint:errcheck
			"status": "accepted",
			"job_id": jobID,
		})
	}
}

func jsonError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"status": "error", "message": msg}) //nolint:errcheck
}
