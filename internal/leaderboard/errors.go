package leaderboard

import (
	"errors"
	"fmt"
	"io/fs"
	"regexp"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
)

var pathRe = regexp.MustCompile(`/[^\s"']+`)

// SanitizeMsg strips filesystem paths from an error message before external
// reporting.
func SanitizeMsg(msg string) string {
	return pathRe.ReplaceAllString(msg, "<path>")
}

// ErrorInfo is the error payload sent with a failed leaderboard submission.
type ErrorInfo struct {
	Code    int    `json:"error_code"`
	Type    string `json:"error_type"`
	Message string `json:"error_message"`
}

// envKeywords indicate environment/infrastructure failures.
var envKeywords = []string{
	"cuda", "gpu", "nvidia", "cudnn", "onnxruntime",
	"gcs", "secret manager", "connection", "timeout",
	"no space", "out of memory", "oom",
}

// Classify maps a pipeline error onto the leaderboard error schema.
//
// error_code:
//
//	1 = user-supplied code (preprocessing script / model loading)
//	2 = our eval scripts / dataloader / pipeline
//	3 = environment (CUDA / GPU / GCS / Secret Manager / network)
//
// Ported from _classify_leaderboard_error; exception class names become
// stable Go-side type strings with the same eval-exit-code mapping.
func Classify(err error) ErrorInfo {
	var exitErr *eval.ExitCodeError
	if errors.As(err, &exitErr) {
		switch exitErr.Code {
		case 10:
			return ErrorInfo{1, "EvalUserCodeError", "Preprocessing script failed to load or execute."}
		case 11:
			return ErrorInfo{3, "EvalEnvironmentError", "CUDA/GPU runtime error in evaluation script."}
		default:
			return ErrorInfo{2, "EvalScriptError",
				fmt.Sprintf("Evaluation script exited with code %d.", exitErr.Code)}
		}
	}

	msg := err.Error()
	lower := strings.ToLower(msg)
	for _, kw := range envKeywords {
		if strings.Contains(lower, kw) {
			return ErrorInfo{3, "EnvironmentError", SanitizeMsg(msg)}
		}
	}

	if errors.Is(err, fs.ErrNotExist) {
		return ErrorInfo{3, "FileNotFoundError", SanitizeMsg(msg)}
	}
	if errors.Is(err, fs.ErrPermission) {
		return ErrorInfo{3, "PermissionError", SanitizeMsg(msg)}
	}

	// Default: our pipeline.
	return ErrorInfo{2, "PipelineError", SanitizeMsg(msg)}
}
