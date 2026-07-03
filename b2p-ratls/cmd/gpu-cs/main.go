package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"os"

	"github.com/yourorg/ratls/pkg/ratls"
)

func main() {
	audience     := getEnv("RATLS_AUDIENCE",  "ratls-buffer-tee")
	listenAddr   := getEnv("LISTEN_ADDR",     ":443")
	internalAddr := getEnv("INTERNAL_ADDR",   "127.0.0.1:8081")

	log.Printf("gpu-cs: starting (audience=%s)", audience)

	srv, err := ratls.NewServer(audience)
	if err != nil {
		log.Fatalf("gpu-cs: init server: %v", err)
	}

	ks := ratls.NewKeyStore()

	// ── External TLS server — reachable by Buffer TEE after RA-TLS ──────────
	mux := http.NewServeMux()
	mux.Handle(ratls.ConnectPath, srv.ConnectHandler())
	mux.HandleFunc("/api/load-model", handleLoadModel)
	mux.HandleFunc("/api/store-key",  handleStoreKey(ks))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	httpSrv := &http.Server{
		Addr:      listenAddr,
		Handler:   mux,
		TLSConfig: srv.TLSConfig(),
	}

	// ── Internal plaintext server — loopback only, for processing manager ───
	// Processing manager calls GET /api/get-key?job_id=<id> to retrieve a key
	// that was deposited by the Buffer TEE via /api/store-key above.
	internalMux := http.NewServeMux()
	internalMux.HandleFunc("/api/get-key", handleGetKey(ks))
	internalSrv := &http.Server{
		Addr:    internalAddr,
		Handler: internalMux,
	}
	go func() {
		ln, err := net.Listen("tcp", internalAddr)
		if err != nil {
			log.Fatalf("gpu-cs: internal listener %s: %v", internalAddr, err)
		}
		log.Printf("gpu-cs: internal key API on %s (loopback only)", internalAddr)
		if err := internalSrv.Serve(ln); err != nil {
			log.Fatalf("gpu-cs: internal server error: %v", err)
		}
	}()

	log.Printf("gpu-cs: listening on %s", listenAddr)
	if err := httpSrv.ListenAndServeTLS("", ""); err != nil {
		log.Fatalf("gpu-cs: server error: %v", err)
	}
}

// handleStoreKey stores a per-job AES-256 symmetric key sent by the Buffer TEE.
//
// POST /api/store-key
// Body: { "job_id": "<job id>", "key_b64": "<standard-base64 AES-256 key>" }
// Response: { "status": "stored", "job_id": "<job id>" }
//
// This endpoint is on the RA-TLS-authenticated TLS server, so only a verified
// Buffer TEE (whose OIDC token passed RA-TLS) can reach it.
func handleStoreKey(ks *ratls.KeyStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			JobID  string `json:"job_id"`
			KeyB64 string `json:"key_b64"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON body", http.StatusBadRequest)
			return
		}
		if req.JobID == "" || req.KeyB64 == "" {
			http.Error(w, "job_id and key_b64 are required", http.StatusBadRequest)
			return
		}
		if err := ks.StoreB64(req.JobID, req.KeyB64); err != nil {
			log.Printf("gpu-cs: store-key %s: %v", req.JobID, err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		log.Printf("gpu-cs: key stored for job_id=%s", req.JobID)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"status": "stored",
			"job_id": req.JobID,
		}); err != nil {
			log.Printf("gpu-cs: encode store-key response: %v", err)
		}
	}
}

// handleGetKey retrieves a stored key by job_id.
// Only served on the loopback INTERNAL_ADDR — never exposed externally.
//
// GET  /api/get-key?job_id=<id>  → { "job_id": "...", "key_b64": "..." }
// DELETE /api/get-key?job_id=<id> → 204 No Content (zeroes and removes key)
func handleGetKey(ks *ratls.KeyStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jobID := r.URL.Query().Get("job_id")
		if jobID == "" {
			http.Error(w, "job_id query param required", http.StatusBadRequest)
			return
		}
		switch r.Method {
		case http.MethodGet:
			keyB64, ok := ks.GetB64(jobID)
			if !ok {
				http.Error(w, "key not found", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(map[string]string{
				"job_id":  jobID,
				"key_b64": keyB64,
			}); err != nil {
				log.Printf("gpu-cs: encode get-key response: %v", err)
			}
		case http.MethodDelete:
			ks.Delete(jobID)
			log.Printf("gpu-cs: key deleted for job_id=%s", jobID)
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func handleLoadModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("gpu-cs: read payload error: %v", err)
		http.Error(w, "read payload failed", http.StatusBadRequest)
		return
	}
	log.Printf("gpu-cs: load model request from %s (%d bytes)", r.RemoteAddr, len(bodyBytes))

	managerURL := getEnv("PROCESSING_MANAGER_JOB_URL", "http://127.0.0.1:4000/enclave/cvm/secure-job")
	resp, err := http.Post(managerURL, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		log.Printf("gpu-cs: forward to processing manager failed: %v", err)
		http.Error(w, "processing manager unavailable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	responseBytes, _ := io.ReadAll(resp.Body)
	log.Printf("gpu-cs: processing manager response status=%d body=%s", resp.StatusCode, string(responseBytes))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(responseBytes); err != nil {
		log.Printf("gpu-cs: write response failed: %v", err)
	}
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
