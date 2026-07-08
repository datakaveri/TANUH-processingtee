// processing-tee is the single Processing TEE binary. It owns the entire
// job pipeline in-process: RA-TLS intake, dataset fetch + decrypt, the
// Python evaluation subprocess (the only Python left at runtime),
// leaderboard submission, the buffer completion callback, and idle/after-job
// self-deallocation. Boot order: startup policy attestation first — the
// server never listens if the enforced network policy cannot be attested.
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/pipeline"
	"github.com/datakaveri/tanuh-processing-tee/internal/policy"
	"github.com/datakaveri/tanuh-processing-tee/internal/ratls"
	"github.com/datakaveri/tanuh-processing-tee/internal/server"
)

func main() {
	cfg := config.FromEnv()

	log.Println("============================================================")
	log.Println("TANUH Processing TEE — starting")
	log.Printf("  RA-TLS server: https://%s (audience=%s)", cfg.ListenAddr, cfg.RATLSAudience)
	log.Printf("  Self-dealloc:  %s/%s/%s (idle %s, after-job %v)",
		cfg.StopProject, cfg.StopZone, cfg.StopInstance, cfg.IdleTimeout, cfg.DeallocAfterJob)
	log.Println("============================================================")

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Startup policy attestation before anything listens (fail-fast, same
	// contract as the former Python pre-step under `set -eu`).
	if err := policy.StartupAttestation(ctx, cfg.PolicyPath, cfg.RATLSAudience); err != nil {
		log.Fatalf("processing-tee: startup policy attestation failed: %v", err)
	}

	rs, err := ratls.NewServer(cfg.RATLSAudience)
	if err != nil {
		log.Fatalf("processing-tee: init RA-TLS server: %v", err)
	}

	mgr, err := pipeline.NewManager(cfg)
	if err != nil {
		log.Fatalf("processing-tee: init pipeline manager: %v", err)
	}
	mgr.StartIdleDeallocator(ctx)

	httpSrv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           server.New(rs, mgr, filepath.Join(cfg.WorkflowDir(), "incoming"), ctx),
		TLSConfig:         rs.TLSConfig(),
		ReadTimeout:       10 * time.Minute,
		WriteTimeout:      10 * time.Minute,
		IdleTimeout:       60 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		<-ctx.Done()
		log.Println("processing-tee: shutting down...")
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutCancel()
		if err := httpSrv.Shutdown(shutCtx); err != nil {
			log.Printf("processing-tee: shutdown error: %v", err)
		}
	}()

	log.Printf("processing-tee: listening on %s (TLS 1.3)", cfg.ListenAddr)
	if err := httpSrv.ListenAndServeTLS("", ""); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("processing-tee: server error: %v", err)
	}
	log.Println("processing-tee: stopped")
}
