package crypto

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"testing"
)

// encryptV1 mirrors the browser producer: chunk the plaintext, seal each chunk
// with IV = base_iv + i and AAD "i:total:fileId", concatenate the ciphertexts.
func encryptV1(t *testing.T, dek, baseIV []byte, fileID string, chunkSize int, plaintext []byte) ([]byte, ManifestV1) {
	t.Helper()
	aead, err := newAESGCM(dek, 12)
	if err != nil {
		t.Fatal(err)
	}
	total := (len(plaintext) + chunkSize - 1) / chunkSize
	if total == 0 {
		total = 1
	}
	var ct []byte
	for i := 0; i < total; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(plaintext) {
			end = len(plaintext)
		}
		aad := []byte(fmt.Sprintf("%d:%d:%s", i, total, fileID))
		ct = append(ct, aead.Seal(nil, ivForChunk(baseIV, i), plaintext[start:end], aad)...)
	}
	sum := sha256.Sum256(plaintext)
	m := ManifestV1{
		Schema:          "tanuh-enc-dataset-v1",
		FileID:          fileID,
		Cipher:          "AES-256-GCM",
		ChunkSize:       chunkSize,
		TotalChunks:     total,
		PlaintextSize:   len(plaintext),
		PlaintextSHA256: hex.EncodeToString(sum[:]),
		BaseIV:          base64.StdEncoding.EncodeToString(baseIV),
		AADScheme:       "i:totalChunks:fileId",
	}
	return ct, m
}

func TestDecryptDatasetV1_RoundTrip(t *testing.T) {
	dek := make([]byte, 32)
	baseIV := make([]byte, 12)
	rand.Read(dek)
	rand.Read(baseIV)

	// Multi-chunk with a partial last chunk: 40 bytes over 16-byte chunks -> 3 chunks.
	plaintext := []byte("the quick brown fox jumps over lazy dog!")
	if len(plaintext) != 40 {
		t.Fatalf("test plaintext len = %d, want 40", len(plaintext))
	}
	ct, m := encryptV1(t, dek, baseIV, "file-abc", 16, plaintext)
	if m.TotalChunks != 3 {
		t.Fatalf("total chunks = %d, want 3", m.TotalChunks)
	}

	got, err := DecryptDatasetV1(dek, m, ct)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if !bytes.Equal(got, plaintext) {
		t.Fatalf("round-trip mismatch:\n got %q\nwant %q", got, plaintext)
	}
}

func TestDecryptDatasetV1_SingleChunk(t *testing.T) {
	dek := make([]byte, 32)
	baseIV := make([]byte, 12)
	rand.Read(dek)
	rand.Read(baseIV)
	plaintext := []byte("small payload")
	ct, m := encryptV1(t, dek, baseIV, "f1", 16*1024*1024, plaintext)
	if m.TotalChunks != 1 {
		t.Fatalf("total chunks = %d, want 1", m.TotalChunks)
	}
	got, err := DecryptDatasetV1(dek, m, ct)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if !bytes.Equal(got, plaintext) {
		t.Fatalf("mismatch: got %q want %q", got, plaintext)
	}
}

func TestDecryptDatasetV1_TamperFailsClosed(t *testing.T) {
	dek := make([]byte, 32)
	baseIV := make([]byte, 12)
	rand.Read(dek)
	rand.Read(baseIV)
	plaintext := bytes.Repeat([]byte("A"), 50)
	ct, m := encryptV1(t, dek, baseIV, "f2", 16, plaintext)

	// Flip a ciphertext byte -> GCM auth must fail.
	bad := append([]byte(nil), ct...)
	bad[0] ^= 0xff
	if _, err := DecryptDatasetV1(dek, m, bad); err == nil {
		t.Fatal("expected authentication failure on tampered ciphertext, got nil")
	}

	// Wrong SHA-256 in the manifest -> integrity check must fail.
	m2 := m
	m2.PlaintextSHA256 = hex.EncodeToString(make([]byte, 32))
	if _, err := DecryptDatasetV1(dek, m2, ct); err == nil {
		t.Fatal("expected SHA-256 mismatch error, got nil")
	}

	// Wrong length -> bounds check must fail.
	if _, err := DecryptDatasetV1(dek, m, ct[:len(ct)-1]); err == nil {
		t.Fatal("expected ciphertext length error, got nil")
	}
}
