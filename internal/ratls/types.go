// Package ratls implements the Processing TEE's server side of RA-TLS:
// a self-signed TLS 1.3 endpoint whose sessions are authenticated by a
// Confidential Space attestation token channel-bound via EKM
// (eat_nonce == base64url(sha256(EKM))).
//
// This package is the attestation audit surface — it must stay free of
// external dependencies.
package ratls

const (
	EKMLabel    = "EXPORTER-ratls/v1"
	EKMLength   = 32
	ConnectPath = "/ratls/connect"
	// MetadataTokenEndpoint is served by the CS launcher via Unix socket.
	// The socket is mounted at /run/container_launcher/teeserver.sock.
	MetadataTokenEndpoint   = "http://localhost/v1/token"
	MetadataTokenSocketPath = "/run/container_launcher/teeserver.sock"
)

// AttestationBundle is what the Processing TEE returns over /ratls/connect.
//
// Flow on this server:
//  1. TLS handshake already complete
//  2. ekm = ExportKeyingMaterial("EXPORTER-ratls/v1", nil, 32)
//  3. nonce = base64url(sha256(ekm))
//  4. OIDC token: POST localhost/v1/token {audience, nonces:[nonce], token_type:"OIDC"}
//  5. Return this bundle: { oidc_token }
//
// The Buffer TEE client independently derives the same EKM from its side of
// the session and verifies the token (Google JWKS; iss/aud/exp/hwmodel/
// image_digest/swname) plus eat_nonce == base64url(sha256(ekm)).
type AttestationBundle struct {
	OIDCToken string `json:"oidc_token"`
}

// TokenRequestBody is the JSON body for POST to MetadataTokenEndpoint.
type TokenRequestBody struct {
	Audience  string   `json:"audience"`
	Nonces    []string `json:"nonces"`
	TokenType string   `json:"token_type"`
}
