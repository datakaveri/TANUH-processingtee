// Package eval runs the dataset-specific Python evaluation script as a
// subprocess. This is the only Python in the pipeline: the scripts are
// fetched from the eval-scripts bucket at job time and invoked with the
// frozen CLI contract:
//
//	python3 <script> --model M --dataset D --results R [--preprocessing P]
//
// Exit codes are part of that contract: 10 = user preprocessing failure,
// 11 = CUDA/GPU environment failure, anything else non-zero = script error.
package eval

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"
)

// ExitCodeError reports a non-zero eval script exit.
type ExitCodeError struct {
	Code int
}

func (e *ExitCodeError) Error() string {
	return fmt.Sprintf("Evaluation script failed with exit code %d", e.Code)
}

// Run executes the evaluation script. Output streams to our stdout/stderr
// (the serial log). timeout of 0 disables the deadline.
func Run(ctx context.Context, workdir, script, model, dataset, results, preprocessing string, timeout time.Duration) error {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	args := []string{script, "--model", model, "--dataset", dataset, "--results", results}
	if preprocessing != "" {
		if _, err := os.Stat(preprocessing); err == nil {
			args = append(args, "--preprocessing", preprocessing)
		}
	}

	cmd := exec.CommandContext(ctx, "python3", args...)
	cmd.Dir = workdir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	log.Printf("eval: running python3 %s", script)
	err := cmd.Run()
	if err == nil {
		return nil
	}
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("eval: evaluation script timed out after %s", timeout)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return &ExitCodeError{Code: exitErr.ExitCode()}
	}
	return fmt.Errorf("eval: run evaluation script: %w", err)
}
