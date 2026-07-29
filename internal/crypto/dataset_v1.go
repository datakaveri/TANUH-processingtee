// dataset_v1.go adds decryption for the "tanuh-enc-dataset-v1" envelope: a
// random AES-256 DEK (wrapped by an asymmetric KMS key, unwrapped separately via
// gcp.AsymmetricDecrypt) encrypts the file in fixed-size chunks. The ciphertext
// is the raw concatenation of chunks (each = plaintext + 16-byte GCM tag, no
// framing); per-chunk IV = base_iv + chunk_index (96-bit big-endian) and AAD =
// "{i}:{totalChunks}:{fileId}". This is the browser-produced format from ui-dx
// encrypted-dataset-upload.ts, distinct from the DecryptBlob/DecryptChunkedFile
// formats in aesgcm.go (which are unchanged).
package crypto

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
)

const gcmTagSize = 16

// ManifestV1 is the <object>.manifest.json sidecar for a tanuh-enc-dataset-v1
// object. JSON tags match the browser producer verbatim.
type ManifestV1 struct {
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

// DecryptDatasetV1 decrypts a tanuh-enc-dataset-v1 ciphertext given the unwrapped
// 32-byte DEK and its manifest. It authenticates every chunk (AAD-bound, so
// reordering/truncation/substitution fail) and verifies the whole-file plaintext
// SHA-256, failing closed on any mismatch.
func DecryptDatasetV1(dek []byte, m ManifestV1, ciphertext []byte) ([]byte, error) {
	baseIV, err := base64.StdEncoding.DecodeString(m.BaseIV)
	if err != nil {
		return nil, fmt.Errorf("crypto: base_iv not base64: %w", err)
	}
	if len(baseIV) != 12 {
		return nil, fmt.Errorf("crypto: base_iv is %d bytes, want 12", len(baseIV))
	}
	if m.TotalChunks < 1 {
		return nil, fmt.Errorf("crypto: total_chunks is %d, want >= 1", m.TotalChunks)
	}
	want := m.PlaintextSize + m.TotalChunks*gcmTagSize
	if len(ciphertext) != want {
		return nil, fmt.Errorf("crypto: ciphertext is %d bytes, expected %d (plaintext_size + %d*chunks)",
			len(ciphertext), want, gcmTagSize)
	}

	aead, err := newAESGCM(dek, 12)
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
			return nil, fmt.Errorf("crypto: negative plaintext length for chunk %d", i)
		}
		end := off + plainLen + gcmTagSize
		if end > len(ciphertext) {
			return nil, fmt.Errorf("crypto: chunk %d exceeds ciphertext bounds", i)
		}
		aad := []byte(fmt.Sprintf("%d:%d:%s", i, m.TotalChunks, m.FileID))
		pt, err := aead.Open(nil, ivForChunk(baseIV, i), ciphertext[off:end], aad)
		if err != nil {
			return nil, fmt.Errorf("crypto: chunk %d/%d authentication failed (corrupt, reordered, or wrong key)",
				i+1, m.TotalChunks)
		}
		off = end
		h.Write(pt)
		out = append(out, pt...)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.PlaintextSHA256 {
		return nil, fmt.Errorf("crypto: plaintext SHA-256 mismatch: manifest=%s decrypted=%s", m.PlaintextSHA256, got)
	}
	return out, nil
}

// ivForChunk returns base_iv + i as a 96-bit big-endian counter (add with carry).
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
