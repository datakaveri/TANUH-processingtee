package gcp

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// AsymmetricDecrypt unwraps a base64-encoded wrapped DEK via Cloud KMS
// asymmetricDecrypt on the given crypto-key VERSION resource name
// (projects/.../cryptoKeyVersions/N), returning the raw key bytes. Uses the
// attached service-account token, so the enclave identity must hold
// cloudkms.cryptoKeyDecrypter on the key.
func AsymmetricDecrypt(ctx context.Context, keyVersion, wrappedDEKB64 string) ([]byte, error) {
	token, err := AccessToken(ctx)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("https://cloudkms.googleapis.com/v1/%s:asymmetricDecrypt", keyVersion)
	body, _ := json.Marshal(map[string]string{"ciphertext": wrappedDEKB64})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gcp: KMS asymmetricDecrypt: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gcp: KMS asymmetricDecrypt: status %d: %s", resp.StatusCode, string(raw))
	}
	var out struct {
		Plaintext string `json:"plaintext"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("gcp: parse KMS response: %w", err)
	}
	dek, err := base64.StdEncoding.DecodeString(out.Plaintext)
	if err != nil {
		return nil, fmt.Errorf("gcp: decode KMS plaintext: %w", err)
	}
	return dek, nil
}
