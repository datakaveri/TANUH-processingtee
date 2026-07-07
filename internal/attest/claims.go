// Package attest fetches this workload's own Confidential Space attestation
// claims (hwmodel, swname, image_digest, secboot, iss) for external
// reporting. The raw JWT is never forwarded anywhere: it is fetched from the
// CS launcher's token server (unix socket), decoded locally, and only those
// five claim fields are included in the leaderboard submission body — the
// leaderboard entry's proof that the eval ran inside an attested TEE.
// TokenViaSocket is also used by the startup policy attestation (policy hash
// as eat_nonce). The metadata-server identity token is a degraded off-TEE
// fallback that lacks the CS claims.
package attest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
	"github.com/datakaveri/tanuh-processing-tee/internal/ratls"
)

// Fixed nonce (10–74 chars) — we read claims, not channel-bind, so any valid
// nonce works; it just appears as eat_nonce in the token.
const leaderboardNonce = "tanuh-leaderboard-attestation-nonce"

// Claims are the attestation fields forwarded to the leaderboard.
type Claims struct {
	HWModel        string `json:"hwmodel"`
	SwName         string `json:"swname"`
	ImageDigest    string `json:"image_digest"`
	ImageReference string `json:"image_reference"`
	InstanceID     string `json:"instance_id"`
	Iat            int64  `json:"iat"`
	Secboot        bool   `json:"secboot"`
	Iss            string `json:"iss"`
}

// TokenViaSocket fetches an OIDC attestation token from the CS launcher's
// token server over its unix socket. Errors off-TEE (no socket).
func TokenViaSocket(ctx context.Context, audience string, nonces []string) (string, error) {
	body, err := json.Marshal(ratls.TokenRequestBody{
		Audience:  audience,
		Nonces:    nonces,
		TokenType: "OIDC",
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		ratls.MetadataTokenEndpoint, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", ratls.MetadataTokenSocketPath)
			},
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("attest: teeserver /v1/token: %w", err)
	}
	defer resp.Body.Close()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		return "", err
	}
	token := strings.TrimSpace(buf.String())
	if resp.StatusCode != http.StatusOK {
		snippet := token
		if len(snippet) > 200 {
			snippet = snippet[:200]
		}
		return "", fmt.Errorf("attest: teeserver /v1/token returned %d: %s", resp.StatusCode, snippet)
	}
	return token, nil
}

// DecodeJWTPayload decodes a JWT's payload without verifying the signature.
// (Signature verification is unnecessary here: the token comes straight from
// the local launcher socket and is only decoded for its claim values.)
func DecodeJWTPayload(token string) map[string]any {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil
	}
	padded := parts[1]
	if m := len(padded) % 4; m != 0 {
		padded += strings.Repeat("=", 4-m)
	}
	raw, err := base64.URLEncoding.DecodeString(padded)
	if err != nil {
		return nil
	}
	var claims map[string]any
	if err := json.Unmarshal(raw, &claims); err != nil {
		return nil
	}
	return claims
}

// FetchClaims extracts this workload's CS attestation claims. Falls back to
// the metadata-server identity token off-TEE (those claims come back blank —
// the best we can do outside Confidential Space). Returns zero Claims if no
// token can be obtained at all.
func FetchClaims(ctx context.Context, audience string) Claims {
	token, err := TokenViaSocket(ctx, audience, []string{leaderboardNonce})
	if err != nil {
		log.Printf("attest: CS attestation token unavailable (%v); falling back to identity token", err)
		token, err = gcp.IdentityToken(ctx, audience)
		if err != nil {
			log.Printf("attest: could not fetch any attestation token: %v", err)
			return Claims{}
		}
	}

	claims := DecodeJWTPayload(token)
	if claims == nil {
		log.Printf("attest: attestation token could not be decoded")
		return Claims{}
	}

	out := Claims{}
	out.HWModel, _ = claims["hwmodel"].(string)
	out.SwName, _ = claims["swname"].(string)
	out.Iss, _ = claims["iss"].(string)
	out.InstanceID, _ = claims["sub"].(string)
	out.Secboot, _ = claims["secboot"].(bool)
	if iat, ok := claims["iat"].(float64); ok {
		out.Iat = int64(iat)
	}
	if submods, ok := claims["submods"].(map[string]any); ok {
		if container, ok := submods["container"].(map[string]any); ok {
			out.ImageDigest, _ = container["image_digest"].(string)
			out.ImageReference, _ = container["image_reference"].(string)
		}
	}
	return out
}
