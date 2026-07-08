package eval

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"time"
)

// DepsError reports a failed pre-eval dependency install for a user
// preprocessing script (dep_scanner.py + uv). Output carries the scanner's
// stderr tail for classification/logging; it is sanitised before any
// external reporting.
type DepsError struct {
	Output string
}

func (e *DepsError) Error() string {
	return "installing preprocessing dependencies failed: " + e.Output
}

// InstallDeps runs dep_scanner.py against the user's preprocessing script:
// it AST-scans the imports and uv-installs whatever the image doesn't
// already provide, so the eval subprocess doesn't die on ImportError.
// The install happens before the eval sandbox launches, matching the
// latestv7 design — a failed install fails the job with the uv output
// rather than a confusing ImportError later.
func InstallDeps(ctx context.Context, workdir, scannerPath, scriptPath string, timeout time.Duration) error {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	log.Printf("eval: scanning %s for missing dependencies", scriptPath)
	cmd := exec.CommandContext(ctx, "python3", scannerPath, scriptPath)
	cmd.Dir = workdir

	var stderr bytes.Buffer
	cmd.Stdout = os.Stdout
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)

	err := cmd.Run()
	if err == nil {
		return nil
	}
	if ctx.Err() == context.DeadlineExceeded {
		return &DepsError{Output: fmt.Sprintf("dependency install timed out after %s", timeout)}
	}
	tail := stderr.String()
	if len(tail) > 1024 {
		tail = tail[len(tail)-1024:]
	}
	return &DepsError{Output: tail}
}