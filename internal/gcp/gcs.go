package gcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// ListObjects returns the names of all objects under prefix in bucket
// (following pagination), using the attached service-account token.
func ListObjects(ctx context.Context, bucket, prefix string) ([]string, error) {
	token, err := AccessToken(ctx)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 30 * time.Second}
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
		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("gcp: GCS list gs://%s/%s: %w", bucket, prefix, err)
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("gcp: GCS list gs://%s/%s: status %d: %s",
				bucket, prefix, resp.StatusCode, string(raw))
		}
		var page struct {
			Items []struct {
				Name string `json:"name"`
			} `json:"items"`
			NextPageToken string `json:"nextPageToken"`
		}
		if err := json.Unmarshal(raw, &page); err != nil {
			return nil, fmt.Errorf("gcp: parse GCS list: %w", err)
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

// DownloadObjectBytes downloads a GCS object fully into memory using the
// attached service-account token. Suitable for manifests and dataset objects
// that fit in RAM (the pipeline already materialises datasets in memory).
func DownloadObjectBytes(ctx context.Context, bucket, object string) ([]byte, error) {
	token, err := AccessToken(ctx)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("https://storage.googleapis.com/download/storage/v1/b/%s/o/%s?alt=media",
		url.PathEscape(bucket), url.QueryEscape(object))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gcp: GCS download gs://%s/%s: %w", bucket, object, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("gcp: GCS download gs://%s/%s: status %d: %s",
			bucket, object, resp.StatusCode, string(raw))
	}
	return io.ReadAll(resp.Body)
}
