// Command decrypt-test is a STANDALONE, throwaway harness to prove we can
// decrypt a "tanuh-enc-dataset-v1" dataset end to end against the real Cloud
// KMS key and the file-server-data bucket. It is deliberately self-contained
// (stdlib only, imports nothing from internal/) and separate from the
// production Processing TEE decrypt path — the attestation-gated KMS
// integration is a later, different job.
//
// Auth: it uses a bearer token from GCLOUD_TOKEN, meant to be an impersonated
// service-account token so the decrypt runs as the identity that actually
// holds cloudkms.cryptoKeyDecrypter:
//
//	export GCLOUD_TOKEN=$(gcloud auth print-access-token \
//	  --impersonate-service-account=tanuh-processing-tee@proj-tanuh-benchmark-ptfm.iam.gserviceaccount.com)
//	go run ./tools/decrypt-test -uuid 0e6d81a8-2d06-4ba4-9397-31927f4d6a67
//
// It lists the UUID folder, and for every <object>.manifest.json it finds:
// unwraps the DEK via KMS, chunk-decrypts + authenticates the object, verifies
// the plaintext SHA-256, and writes the plaintext to -out. Exit 0 only if every
// dataset round-trips.
package main

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const gcmTag = 16

type manifestV1 struct {
	Schema          string `json:"schema"`
	FileID          string `json:"file_id"`
	Cipher          string `json:"cipher"`
	ChunkSize       int    `json:"chunk_size"`
	TotalChunks     int    `json:"total_chunks"`
	PlaintextSize   int    `json:"plaintext_size"`
	PlaintextSHA256 string `json:"plaintext_sha256"`
	BaseIV          string `json:"base_iv"`
	AADScheme       string `json:"aad_scheme"`
	WrappedDEK      string `json:"wrapped_dek"`
	DEKWrapAlg      string `json:"dek_wrap_alg"`
	KMSKeyVersion   string `json:"kms_key_version"`
	CreatedAt       string `json:"created_at"`
}

func main() {
	bucket := flag.String("bucket", "file-server-data", "GCS bucket holding the dataset folders")
	uuid := flag.String("uuid", "", "dataset folder UUID (the object prefix)")
	outDir := flag.String("out", "./decrypt-out", "directory to write decrypted plaintext into")
	flag.Parse()

	token := os.Getenv("GCLOUD_TOKEN")
	if token == "" {
		fatal("GCLOUD_TOKEN is empty — set it to an (impersonated) access token, e.g.\n" +
			"  export GCLOUD_TOKEN=$(gcloud auth print-access-token --impersonate-service-account=tanuh-processing-tee@proj-tanuh-benchmark-ptfm.iam.gserviceaccount.com)")
	}
	if *uuid == "" {
		fatal("-uuid is required (the dataset folder UUID)")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err.Error())
	}

	ctx := context.Background()
	prefix := *uuid + "/"

	fmt.Printf("Listing gs://%s/%s ...\n", *bucket, prefix)
	names, err := listObjects(ctx, token, *bucket, prefix)
	if err != nil {
		fatal(err.Error())
	}
	if len(names) == 0 {
		fatal("no objects under that prefix")
	}
	for _, n := range names {
		fmt.Printf("  found: %s\n", n)
	}

	present := make(map[string]bool, len(names))
	for _, n := range names {
		present[n] = true
	}

	var manifests []string
	for _, n := range names {
		if strings.HasSuffix(n, ".manifest.json") {
			manifests = append(manifests, n)
		}
	}
	if len(manifests) == 0 {
		fatal("no *.manifest.json found under the UUID folder")
	}

	failures := 0
	for _, mname := range manifests {
		cipherName := strings.TrimSuffix(mname, ".manifest.json")
		if !present[cipherName] {
			fmt.Printf("\n[SKIP] %s: no matching ciphertext object %q\n", mname, cipherName)
			failures++
			continue
		}
		if err := decryptOne(ctx, token, *bucket, mname, cipherName, *outDir); err != nil {
			fmt.Printf("\n[FAIL] %s: %v\n", cipherName, err)
			failures++
			continue
		}
	}

	fmt.Printf("\n%d/%d dataset(s) decrypted + verified\n", len(manifests)-failures, len(manifests))
	if failures > 0 {
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func decryptOne(ctx context.Context, token, bucket, manifestObj, cipherObj, outDir string) error {
	fmt.Printf("\n=== %s ===\n", cipherObj)

	// 1. manifest
	mbytes, err := downloadBytes(ctx, token, bucket, manifestObj)
	if err != nil {
		return fmt.Errorf("download manifest: %w", err)
	}
	var m manifestV1
	if err := json.Unmarshal(mbytes, &m); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	if m.Schema != "tanuh-enc-dataset-v1" {
		return fmt.Errorf("unsupported manifest schema %q", m.Schema)
	}
	fmt.Printf("manifest: %d chunk(s), %d plaintext bytes, key=%s\n", m.TotalChunks, m.PlaintextSize, m.KMSKeyVersion)

	// 2. unwrap DEK via KMS asymmetricDecrypt
	fmt.Printf("unwrapping DEK via KMS...\n")
	dek, err := kmsAsymmetricDecrypt(ctx, token, m.KMSKeyVersion, m.WrappedDEK)
	if err != nil {
		return fmt.Errorf("KMS unwrap: %w", err)
	}
	if len(dek) != 32 {
		return fmt.Errorf("unwrapped DEK is %d bytes, want 32", len(dek))
	}
	fmt.Printf("DEK unwrapped (32 bytes)\n")

	// 3. ciphertext
	ct, err := downloadBytes(ctx, token, bucket, cipherObj)
	if err != nil {
		return fmt.Errorf("download ciphertext: %w", err)
	}

	// 4. decrypt + verify
	pt, err := decryptDatasetV1(dek, m, ct)
	if err != nil {
		return err
	}

	// 5. write plaintext
	outPath := filepath.Join(outDir, filepath.Base(cipherObj))
	if err := os.WriteFile(outPath, pt, 0o644); err != nil {
		return fmt.Errorf("write plaintext: %w", err)
	}
	fmt.Printf("OK — %d bytes, SHA-256 %s verified → %s\n", len(pt), m.PlaintextSHA256, outPath)
	return nil
}

// decryptDatasetV1 mirrors the reference decrypt-dataset.mjs contract.
func decryptDatasetV1(dek []byte, m manifestV1, ciphertext []byte) ([]byte, error) {
	baseIV, err := base64.StdEncoding.DecodeString(m.BaseIV)
	if err != nil {
		return nil, fmt.Errorf("base_iv not base64: %w", err)
	}
	if len(baseIV) != 12 {
		return nil, fmt.Errorf("base_iv is %d bytes, want 12", len(baseIV))
	}
	want := m.PlaintextSize + m.TotalChunks*gcmTag
	if len(ciphertext) != want {
		return nil, fmt.Errorf("ciphertext is %d bytes, expected %d (plaintext_size + %d*chunks)", len(ciphertext), want, gcmTag)
	}

	block, err := aes.NewCipher(dek)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	h := sha256.New()
	out := make([]byte, 0, m.PlaintextSize)
	off := 0
	for i := 0; i < m.TotalChunks; i++ {
		plainLen := m.ChunkSize
		if i == m.TotalChunks-1 {
			plainLen = m.PlaintextSize - (m.TotalChunks-1)*m.ChunkSize
		}
		if plainLen < 0 {
			return nil, fmt.Errorf("negative plaintext length for chunk %d", i)
		}
		end := off + plainLen + gcmTag
		aad := []byte(fmt.Sprintf("%d:%d:%s", i, m.TotalChunks, m.FileID))
		chunkPT, err := aead.Open(nil, ivForChunk(baseIV, i), ciphertext[off:end], aad)
		if err != nil {
			return nil, fmt.Errorf("chunk %d/%d authentication failed (corrupt, reordered, or wrong key)", i+1, m.TotalChunks)
		}
		off = end
		h.Write(chunkPT)
		out = append(out, chunkPT...)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.PlaintextSHA256 {
		return nil, fmt.Errorf("SHA-256 mismatch: manifest=%s decrypted=%s", m.PlaintextSHA256, got)
	}
	return out, nil
}

// ivForChunk = base_iv + i, 96-bit big-endian with carry.
func ivForChunk(base []byte, i int) []byte {
	iv := make([]byte, len(base))
	copy(iv, base)
	carry := uint64(i)
	for j := len(iv) - 1; j >= 0 && carry > 0; j-- {
		sum := uint64(iv[j]) + (carry & 0xff)
		iv[j] = byte(sum & 0xff)
		carry = (carry >> 8) + (sum >> 8)
	}
	return iv
}

func kmsAsymmetricDecrypt(ctx context.Context, token, keyVersion, wrappedB64 string) ([]byte, error) {
	u := fmt.Sprintf("https://cloudkms.googleapis.com/v1/%s:asymmetricDecrypt", keyVersion)
	body, _ := json.Marshal(map[string]string{"ciphertext": wrappedB64})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(raw))
	}
	var out struct {
		Plaintext string `json:"plaintext"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(out.Plaintext)
}

func listObjects(ctx context.Context, token, bucket, prefix string) ([]string, error) {
	var names []string
	pageToken := ""
	for {
		u := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s",
			url.PathEscape(bucket), url.QueryEscape(prefix))
		if pageToken != "" {
			u += "&pageToken=" + url.QueryEscape(pageToken)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+token)
		resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
		if err != nil {
			return nil, err
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("list status %d: %s", resp.StatusCode, string(raw))
		}
		var page struct {
			Items []struct {
				Name string `json:"name"`
			} `json:"items"`
			NextPageToken string `json:"nextPageToken"`
		}
		if err := json.Unmarshal(raw, &page); err != nil {
			return nil, err
		}
		for _, it := range page.Items {
			names = append(names, it.Name)
		}
		if page.NextPageToken == "" {
			return names, nil
		}
		pageToken = page.NextPageToken
	}
}

func downloadBytes(ctx context.Context, token, bucket, object string) ([]byte, error) {
	u := fmt.Sprintf("https://storage.googleapis.com/download/storage/v1/b/%s/o/%s?alt=media",
		url.PathEscape(bucket), url.QueryEscape(object))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := (&http.Client{Timeout: 10 * time.Minute}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("download %s status %d: %s", object, resp.StatusCode, string(raw))
	}
	return io.ReadAll(resp.Body)
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "error: "+msg)
	os.Exit(2)
}
