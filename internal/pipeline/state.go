// Package pipeline owns the secure-job lifecycle: payload materialisation,
// dataset fetch + decrypt, eval subprocess, leaderboard submission, the
// completion callback to the Buffer TEE, and the idle/after-job
// self-deallocation of this VM. One job runs at a time.
package pipeline

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
)

// State mirrors the former SECURE_JOB_STATE dict; it is persisted to
// runtime/secure_runtime_state.json after every change for debuggability
// (the serial log carries the same updates).
type State struct {
	Status                string `json:"status"`
	LastActivityUnix      int64  `json:"last_activity_unix"`
	CurrentJobID          string `json:"current_job_id"`
	LastJobID             string `json:"last_job_id"`
	LastError             string `json:"last_error"`
	DeallocationRequested bool   `json:"deallocation_requested"`
}

// Manager serialises job execution and owns the runtime state.
type Manager struct {
	cfg config.Config

	mu      sync.Mutex
	state   State
	running bool
}

// NewManager creates the workflow directories and persists the initial state.
func NewManager(cfg config.Config) (*Manager, error) {
	m := &Manager{
		cfg: cfg,
		state: State{
			Status:           "waiting_for_job",
			LastActivityUnix: time.Now().Unix(),
		},
	}
	for _, d := range []string{
		m.workflowDir(), m.artifactsDir(), m.incomingDir(), m.runtimeDir(), m.secureJobsDir(),
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	m.persistState()
	return m, nil
}

func (m *Manager) workflowDir() string   { return m.cfg.WorkflowDir() }
func (m *Manager) artifactsDir() string  { return filepath.Join(m.workflowDir(), "artifacts") }
func (m *Manager) incomingDir() string   { return filepath.Join(m.workflowDir(), "incoming") }
func (m *Manager) runtimeDir() string    { return filepath.Join(m.workflowDir(), "runtime") }
func (m *Manager) secureJobsDir() string { return filepath.Join(m.workflowDir(), "secure_jobs") }
func (m *Manager) jobDir(jobID string) string {
	return filepath.Join(m.secureJobsDir(), jobID)
}
func (m *Manager) statePath() string {
	return filepath.Join(m.runtimeDir(), "secure_runtime_state.json")
}

// setState applies updates under the lock, stamps last_activity, persists,
// and logs — the equivalent of _set_secure_job_state.
func (m *Manager) setState(update func(*State)) State {
	m.mu.Lock()
	defer m.mu.Unlock()
	update(&m.state)
	m.state.LastActivityUnix = time.Now().Unix()
	m.persistState()
	log.Printf("pipeline: state updated: %+v", m.state)
	return m.state
}

// persistState writes the state file. Callers hold the lock (or are in
// single-threaded startup).
func (m *Manager) persistState() {
	raw, err := json.MarshalIndent(m.state, "", "  ")
	if err != nil {
		log.Printf("pipeline: marshal state: %v", err)
		return
	}
	if err := os.WriteFile(m.statePath(), raw, 0o644); err != nil {
		log.Printf("pipeline: persist state: %v", err)
	}
}

// TryStartJob acquires the single job slot. Returns false (and the id of the
// job holding the slot) when a job is already in flight → HTTP 409.
func (m *Manager) TryStartJob(jobID string) (bool, string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.running {
		return false, m.state.CurrentJobID
	}
	m.running = true
	m.state.Status = "job_received"
	m.state.CurrentJobID = jobID
	m.state.LastJobID = jobID
	m.state.LastError = ""
	m.state.DeallocationRequested = false
	m.state.LastActivityUnix = time.Now().Unix()
	m.persistState()
	return true, jobID
}

func (m *Manager) releaseJobSlot() {
	m.mu.Lock()
	m.running = false
	m.mu.Unlock()
}

// RequestDeallocation stops this VM via the Compute API (the former
// stop-processing-vm.sh). Duplicate requests are ignored.
func (m *Manager) RequestDeallocation(ctx context.Context, reason string) {
	m.mu.Lock()
	if m.state.DeallocationRequested {
		m.mu.Unlock()
		log.Printf("pipeline: VM deallocation already requested; skipping duplicate (%s)", reason)
		return
	}
	m.state.DeallocationRequested = true
	m.persistState()
	m.mu.Unlock()

	log.Printf("pipeline: requesting deallocation of %s/%s/%s (%s)",
		m.cfg.StopProject, m.cfg.StopZone, m.cfg.StopInstance, reason)
	if err := gcp.StopInstance(ctx, m.cfg.StopProject, m.cfg.StopZone, m.cfg.StopInstance); err != nil {
		log.Printf("pipeline: VM stop request failed: %v", err)
		m.setState(func(s *State) {
			s.Status = "deallocation_failed"
			s.LastError = "stop request failed: " + err.Error()
		})
		return
	}
	m.setState(func(s *State) {
		s.Status = "deallocation_requested"
		s.LastError = ""
	})
}

// StartIdleDeallocator launches the goroutine that stops the VM after
// IdleTimeout without a job (a live job holds the slot and resets activity).
func (m *Manager) StartIdleDeallocator(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			m.mu.Lock()
			skip := m.state.DeallocationRequested || m.running
			idleFor := time.Since(time.Unix(m.state.LastActivityUnix, 0))
			m.mu.Unlock()

			if skip {
				continue
			}
			if idleFor >= m.cfg.IdleTimeout {
				log.Printf("pipeline: no secure job for %s; requesting VM deallocation", idleFor.Round(time.Second))
				m.RequestDeallocation(ctx,
					"idle timeout exceeded ("+idleFor.Round(time.Second).String()+" >= "+m.cfg.IdleTimeout.String()+")")
			}
		}
	}()
	log.Printf("pipeline: idle deallocator started (timeout=%s)", m.cfg.IdleTimeout)
}
