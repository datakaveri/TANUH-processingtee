// Package policy ports policy/startup_policy_attestation.py: at boot,
// before serving any traffic, the enforced network policy is loaded, hashed
// deterministically, and bound into a Confidential Space attestation token's
// eat_nonce so a verifier can prove which policy was active when the
// workload executed. The evidence is written to the serial log.
package policy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
)

// Rule is one network policy rule (loosely typed: the policy file is the
// source of truth; unknown fields must survive hashing).
type Rule map[string]any

// Load reads the policy JSON file.
func Load(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("policy: read %s: %w", path, err)
	}
	var p map[string]any
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("policy: parse %s: %w", path, err)
	}
	return p, nil
}

// CanonicalJSON serialises the policy exactly like Python's
// json.dumps(policy, sort_keys=True, separators=(",", ":")) for ASCII
// content: sorted keys, no whitespace, no HTML escaping. The startup hash
// must stay stable across the Python → Go port — see the golden test.
func CanonicalJSON(v any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	// Encoder appends a trailing newline; Python's dumps does not.
	return strings.TrimSuffix(buf.String(), "\n"), nil
}

// Hash returns the SHA-256 hex of the canonical policy JSON.
func Hash(p map[string]any) (string, error) {
	canonical, err := CanonicalJSON(p)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(canonical))
	return hex.EncodeToString(sum[:]), nil
}

// StartupAttestation performs the boot-time policy attestation: log the
// enforced rules, compute the policy hash, fetch a CS attestation token with
// the hash as eat_nonce, and verify the hash round-tripped. Returns an error
// if the token cannot be fetched or the nonce does not match — the caller
// treats that as fatal, exactly like the former Python pre-step under
// `set -eu`.
func StartupAttestation(ctx context.Context, policyPath, audience string) error {
	log.Println("================================================")
	log.Println(" POLICY STARTUP ATTESTATION")
	log.Println("================================================")

	p, err := Load(policyPath)
	if err != nil {
		return err
	}

	rules, _ := p["rules"].([]any)
	log.Printf("ENFORCED NETWORK RULES (%d):", len(rules))
	for i, r := range rules {
		rule, _ := r.(map[string]any)
		dest := rule["destination"]
		if dest == nil {
			dest = rule["destinations"]
		}
		log.Printf("  Rule %d: dest=%v protocol=%v ports=%v download=%v upload=%v — %v",
			i+1, dest, rule["protocol"], rule["ports"],
			orNA(rule["download"]), orNA(rule["upload"]), rule["description"])
	}

	hash, err := Hash(p)
	if err != nil {
		return err
	}
	log.Printf("Policy SHA256: %s", hash)

	log.Println("Requesting startup JWT with policy hash as nonce...")
	token, err := attest.TokenViaSocket(ctx, audience, []string{hash})
	if err != nil {
		return fmt.Errorf("policy: startup attestation token: %w", err)
	}

	claims := attest.DecodeJWTPayload(token)
	if claims == nil {
		return fmt.Errorf("policy: startup attestation token could not be decoded")
	}

	nonce := fmt.Sprintf("%v", claims["eat_nonce"])
	log.Printf("EAT NONCE: %s", nonce)
	if !strings.Contains(nonce, hash) {
		return fmt.Errorf("policy: FAIL — policy hash missing from eat_nonce")
	}
	log.Println("PASS: Policy hash verified inside JWT")
	log.Println("================================================")
	log.Println(" Startup Policy Attestation Complete")
	log.Println("================================================")
	return nil
}

func orNA(v any) any {
	if v == nil {
		return "N/A"
	}
	return v
}
