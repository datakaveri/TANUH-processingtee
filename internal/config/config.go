// Package config centralises all runtime configuration for the Processing
// TEE. Everything is parsed once in main from the environment (the Dockerfile
// is the single source of default values for deployment); the dataset table
// mirrors the former config.yml gcp.dataset_map.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Dataset holds the GCS/Secret Manager coordinates for one dataset.
type Dataset struct {
	JSONObject       string // AES-GCM encrypted dataset JSON (paths + labels)
	ImagesObject     string // AES-GCM encrypted zip of image files
	SecretID         string // Secret Manager secret holding the AES-256 key (hex)
	EvalScriptObject string // eval script object in the eval-scripts bucket
	ImagesExtractDir string // where images are extracted inside the TEE
}

// Config is the full runtime configuration.
type Config struct {
	BaseDir       string
	ListenAddr    string
	RATLSAudience string

	ProjectID         string
	DatasetsBucket    string
	EvalScriptsBucket string
	Datasets          map[int]Dataset

	// Self-deallocation target (this VM). INSTANCE must be overridden per VM
	// (gpu-cs-tdx-h100 / cpu-cs-tdx) via tee-env metadata.
	StopProject  string
	StopZone     string
	StopInstance string

	IdleTimeout       time.Duration
	DeallocAfterJob   bool
	EvalTimeout       time.Duration // 0 disables the timeout
	LeaderboardURL    string
	PolicyPath        string
	AttestAudience    string // audience for leaderboard attestation claims
	CallbackAudience  string // audience for the buffer completion-callback token
	CallbackTimeout   time.Duration
}

// FromEnv builds the Config from the environment.
func FromEnv() Config {
	base := getEnv("BASE_DIR", "/app")
	return Config{
		BaseDir:       base,
		ListenAddr:    getEnv("LISTEN_ADDR", ":443"),
		RATLSAudience: getEnv("RATLS_AUDIENCE", "ratls-buffer-tee"),

		ProjectID:         getEnv("GCP_PROJECT_ID", "p3dx-depa-sandbox"),
		DatasetsBucket:    getEnv("DATASETS_BUCKET", "tanuh-datasets"),
		EvalScriptsBucket: getEnv("EVAL_SCRIPTS_BUCKET", "tanuh-eval-scripts"),
		Datasets: map[int]Dataset{
			1: {
				JSONObject:       "breast-cancer/dataset.json.enc",
				ImagesObject:     "breast-cancer/images.zip.enc",
				SecretID:         "tanuh-ds1-key",
				EvalScriptObject: "evaluate_model_breastcancer.py",
				ImagesExtractDir: base + "/cvm_workflow/data/breast-cancer",
			},
			2: {
				JSONObject:       "ocs/dataset.json.enc",
				ImagesObject:     "ocs/images.zip.enc",
				SecretID:         "tanuh-ds2-key",
				EvalScriptObject: "evaluate_model_OCS.py",
				ImagesExtractDir: base + "/cvm_workflow/data/ocs",
			},
		},

		StopProject:  getEnv("PROJECT", "p3dx-depa-sandbox"),
		StopZone:     getEnv("ZONE", "us-central1-a"),
		StopInstance: getEnv("INSTANCE", "gpu-cs-tdx-h100"),

		IdleTimeout:     secondsEnv("PROCESSING_IDLE_TIMEOUT_SECONDS", 300),
		DeallocAfterJob: getEnv("PROCESSING_DEALLOCATE_AFTER_JOB", "1") == "1",
		EvalTimeout:     secondsEnv("PROCESSING_EVAL_TIMEOUT_SECONDS", 3600),
		LeaderboardURL: getEnv("LEADERBOARD_SUBMIT_URL",
			"https://benchmark.tanuh.ai/leaderboard/submit-solution"),
		PolicyPath:       getEnv("NETWORK_POLICY_PATH", base+"/policy/network_policy.json"),
		AttestAudience:   "https://tanuh-processing-tee",
		CallbackAudience: getEnv("CALLBACK_AUDIENCE", "tanuh-buffer-callback"),
		CallbackTimeout:  10 * time.Second,
	}
}

// WorkflowDir returns BASE_DIR/cvm_workflow.
func (c Config) WorkflowDir() string { return c.BaseDir + "/cvm_workflow" }

// Dataset returns the coordinates for datasetID or an error for unknown ids.
func (c Config) Dataset(datasetID int) (Dataset, error) {
	d, ok := c.Datasets[datasetID]
	if !ok {
		return Dataset{}, fmt.Errorf("config: no dataset configured for dataset_id=%d", datasetID)
	}
	return d, nil
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func secondsEnv(key string, def int) time.Duration {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return time.Duration(n) * time.Second
		}
	}
	return time.Duration(def) * time.Second
}
