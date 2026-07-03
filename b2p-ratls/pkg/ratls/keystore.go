package ratls

import (
	"encoding/base64"
	"fmt"
	"sync"
)

// KeyStore is a thread-safe in-memory map of job_id → raw AES-256 key bytes.
// Keys are deposited by the Buffer TEE (POST /api/store-key) after RA-TLS
// verification and retrieved by the local processing manager (GET /api/get-key).
type KeyStore struct {
	mu   sync.RWMutex
	keys map[string][]byte
}

// NewKeyStore returns an empty KeyStore.
func NewKeyStore() *KeyStore {
	return &KeyStore{keys: make(map[string][]byte)}
}

// StoreB64 decodes standard-base64 (or base64url-no-pad) keyB64 and stores
// it under jobID. Returns an error if the decoded key is not exactly 32 bytes.
func (ks *KeyStore) StoreB64(jobID, keyB64 string) error {
	raw, err := base64.StdEncoding.DecodeString(keyB64)
	if err != nil {
		raw, err = base64.URLEncoding.WithPadding(base64.NoPadding).DecodeString(keyB64)
		if err != nil {
			return fmt.Errorf("key_b64 is not valid base64: %w", err)
		}
	}
	if len(raw) != 32 {
		return fmt.Errorf("expected 32-byte AES-256 key, got %d bytes", len(raw))
	}
	cp := make([]byte, 32)
	copy(cp, raw)

	ks.mu.Lock()
	ks.keys[jobID] = cp
	ks.mu.Unlock()
	return nil
}

// GetB64 returns the standard-base64-encoded key for jobID.
// Returns ("", false) if no key has been stored for that job.
func (ks *KeyStore) GetB64(jobID string) (string, bool) {
	ks.mu.RLock()
	raw, ok := ks.keys[jobID]
	ks.mu.RUnlock()
	if !ok {
		return "", false
	}
	return base64.StdEncoding.EncodeToString(raw), true
}

// Delete zeros the stored bytes and removes the entry for jobID.
// Call this after the key has been consumed to limit its in-memory lifetime.
func (ks *KeyStore) Delete(jobID string) {
	ks.mu.Lock()
	defer ks.mu.Unlock()
	if raw, ok := ks.keys[jobID]; ok {
		for i := range raw {
			raw[i] = 0
		}
		delete(ks.keys, jobID)
	}
}
