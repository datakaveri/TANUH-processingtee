package pipeline

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
)

// decryptDatasetJSON downloads the encrypted dataset JSON, decrypts it in
// memory (it is small), validates it parses, and stages the plaintext in the
// job's runtime dir — the path handed to the eval script.
func (m *Manager) decryptDatasetJSON(ctx context.Context, key []byte, ds config.Dataset, job *materialized) (string, error) {
	encPath := filepath.Join(m.artifactsDir(), fmt.Sprintf("dataset_%d.json.enc", job.datasetID))
	log.Printf("pipeline: downloading encrypted dataset JSON gs://%s/%s", m.cfg.DatasetsBucket, ds.JSONObject)
	if err := gcp.DownloadObject(ctx, m.cfg.DatasetsBucket, ds.JSONObject, encPath); err != nil {
		return "", err
	}
	encData, err := os.ReadFile(encPath)
	if err != nil {
		return "", err
	}
	_ = os.Remove(encPath)

	plaintext, err := crypto.DecryptBlob(key, encData)
	if err != nil {
		return "", fmt.Errorf("pipeline: decrypt dataset JSON: %w", err)
	}
	if !json.Valid(plaintext) {
		return "", fmt.Errorf("pipeline: decrypted dataset is not valid JSON")
	}

	datasetPath := filepath.Join(job.runtimeDir, fmt.Sprintf("dataset_%d_decrypted.json", job.datasetID))
	if err := os.WriteFile(datasetPath, plaintext, 0o644); err != nil {
		return "", err
	}
	return datasetPath, nil
}

// fetchAndExtractImages downloads the encrypted image zip, decrypts it
// (streaming AES-GCM chunks), and extracts it to the dataset's extract dir.
func (m *Manager) fetchAndExtractImages(ctx context.Context, key []byte, ds config.Dataset, datasetID int) error {
	encZipPath := filepath.Join(m.artifactsDir(), fmt.Sprintf("dataset_%d_images.zip.enc", datasetID))
	log.Printf("pipeline: downloading encrypted images gs://%s/%s", m.cfg.DatasetsBucket, ds.ImagesObject)
	if err := gcp.DownloadObject(ctx, m.cfg.DatasetsBucket, ds.ImagesObject, encZipPath); err != nil {
		return err
	}

	zipPath := filepath.Join(m.artifactsDir(), fmt.Sprintf("dataset_%d_images.zip", datasetID))
	log.Printf("pipeline: decrypting image zip...")
	if err := crypto.DecryptChunkedFile(key, encZipPath, zipPath); err != nil {
		return fmt.Errorf("pipeline: decrypt image zip: %w", err)
	}
	_ = os.Remove(encZipPath)

	log.Printf("pipeline: extracting images to %s...", ds.ImagesExtractDir)
	if err := unzip(zipPath, ds.ImagesExtractDir); err != nil {
		return fmt.Errorf("pipeline: extract images: %w", err)
	}
	_ = os.Remove(zipPath)

	log.Printf("pipeline: images extracted to %s", ds.ImagesExtractDir)
	return nil
}

// unzip extracts an archive with zip-slip protection.
func unzip(zipPath, destDir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	cleanDest := filepath.Clean(destDir)

	for _, f := range r.File {
		target := filepath.Join(cleanDest, f.Name)
		if !strings.HasPrefix(filepath.Clean(target), cleanDest+string(os.PathSeparator)) {
			return fmt.Errorf("zip entry escapes destination: %q", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		if err := extractFile(f, target); err != nil {
			return fmt.Errorf("extract %s: %w", f.Name, err)
		}
	}
	return nil
}

func extractFile(f *zip.File, target string) error {
	src, err := f.Open()
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(target)
	if err != nil {
		return err
	}
	defer dst.Close()
	_, err = io.Copy(dst, src)
	return err
}
