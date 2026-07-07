// Package gcp provides the minimal GCP REST calls the Processing TEE needs
// (metadata tokens, GCS object download, Secret Manager access, Compute stop),
// implemented with stdlib net/http only — no Cloud SDK dependency, keeping
// go.mod empty and the audit surface small.
package gcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"
)

const metadataRoot = "http://metadata.google.internal/computeMetadata/v1"

// AccessToken returns the OAuth2 access token for the attached service
// account from the metadata server. The metadata server transparently
// enforces WIF attribute conditions before issuing this token.
func AccessToken(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		metadataRoot+"/instance/service-accounts/default/token", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("gcp: metadata token: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("gcp: metadata token: status %d", resp.StatusCode)
	}

	var body struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", fmt.Errorf("gcp: parse metadata token: %w", err)
	}
	if body.AccessToken == "" {
		return "", fmt.Errorf("gcp: metadata token empty")
	}
	return body.AccessToken, nil
}

// IdentityToken returns the OIDC identity token for the attached service
// account (format=full). Used only as the off-TEE fallback for attestation
// claims — it lacks the Confidential Space claims.
func IdentityToken(ctx context.Context, audience string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		metadataRoot+"/instance/service-accounts/default/identity?audience="+
			url.QueryEscape(audience)+"&format=full", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("gcp: identity token: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("gcp: identity token: status %d", resp.StatusCode)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// DownloadObject streams a GCS object to destPath using the instance token.
func DownloadObject(ctx context.Context, bucket, object, destPath string) error {
	token, err := AccessToken(ctx)
	if err != nil {
		return err
	}
	u := fmt.Sprintf("https://storage.googleapis.com/download/storage/v1/b/%s/o/%s?alt=media",
		bucket, url.QueryEscape(object))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("gcp: GCS download gs://%s/%s: %w", bucket, object, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("gcp: GCS download gs://%s/%s: status %d: %s",
			bucket, object, resp.StatusCode, string(body))
	}

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		return fmt.Errorf("gcp: write %s: %w", destPath, err)
	}
	return out.Sync()
}

// AccessSecret fetches the latest version of a Secret Manager secret's
// payload bytes, retrying up to 3 times on transient errors.
func AccessSecret(ctx context.Context, projectID, secretID string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(1<<(attempt-1)) * time.Second):
			}
		}
		raw, err := accessSecretOnce(ctx, projectID, secretID)
		if err == nil {
			return raw, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

func accessSecretOnce(ctx context.Context, projectID, secretID string) ([]byte, error) {
	token, err := AccessToken(ctx)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("https://secretmanager.googleapis.com/v1/projects/%s/secrets/%s/versions/latest:access",
		projectID, secretID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gcp: secret %s: %w", secretID, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("gcp: secret %s: status %d: %s", secretID, resp.StatusCode, string(body))
	}
	var payload struct {
		Payload struct {
			Data string `json:"data"`
		} `json:"payload"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("gcp: parse secret %s: %w", secretID, err)
	}
	raw, err := base64.StdEncoding.DecodeString(payload.Payload.Data)
	if err != nil {
		return nil, fmt.Errorf("gcp: decode secret %s: %w", secretID, err)
	}
	return raw, nil
}

// StopInstance requests a Compute Engine stop of the given instance
// (?discardLocalSsd=true, matching the former stop-processing-vm.sh).
// A 200/202 response means the stop was accepted.
func StopInstance(ctx context.Context, project, zone, instance string) error {
	token, err := AccessToken(ctx)
	if err != nil {
		return err
	}
	u := fmt.Sprintf("https://compute.googleapis.com/compute/v1/projects/%s/zones/%s/instances/%s/stop?discardLocalSsd=true",
		project, zone, instance)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("gcp: stop %s: %w", instance, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return fmt.Errorf("gcp: stop %s: status %d: %s", instance, resp.StatusCode, string(body))
	}
	return nil
}
