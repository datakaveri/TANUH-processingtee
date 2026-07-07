package crypto

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// encryptBlob mirrors the dataset tooling's small-blob format:
// [4B nonce_len][nonce][ciphertext+tag]
func encryptBlob(t *testing.T, key, plaintext []byte) []byte {
	t.Helper()
	block, _ := aes.NewCipher(key)
	aead, _ := cipher.NewGCM(block)
	nonce := make([]byte, 12)
	if _, err := rand.Read(nonce); err != nil {
		t.Fatal(err)
	}
	ct := aead.Seal(nil, nonce, plaintext, nil)

	var buf bytes.Buffer
	binary.Write(&buf, binary.BigEndian, uint32(len(nonce))) //nolint:errcheck
	buf.Write(nonce)
	buf.Write(ct)
	return buf.Bytes()
}

func TestDecryptBlobRoundtrip(t *testing.T) {
	key := bytes.Repeat([]byte{7}, 32)
	plaintext := []byte(`{"dataset":"ok"}`)

	got, err := DecryptBlob(key, encryptBlob(t, key, plaintext))
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if !bytes.Equal(got, plaintext) {
		t.Fatalf("roundtrip mismatch: %q", got)
	}
}

func TestDecryptBlobRejectsTamper(t *testing.T) {
	key := bytes.Repeat([]byte{7}, 32)
	enc := encryptBlob(t, key, []byte("data"))
	enc[len(enc)-1] ^= 1
	if _, err := DecryptBlob(key, enc); err == nil {
		t.Fatal("tampered blob decrypted successfully")
	}
}

// TestDecryptChunkedFileRoundtrip mirrors the chunked format:
// [4B num_chunks] then per chunk [4B len(nonce+ct)][nonce(12)][ct+tag]
func TestDecryptChunkedFileRoundtrip(t *testing.T) {
	key := bytes.Repeat([]byte{3}, 32)
	block, _ := aes.NewCipher(key)
	aead, _ := cipher.NewGCM(block)

	chunks := [][]byte{
		bytes.Repeat([]byte("A"), 1000),
		bytes.Repeat([]byte("B"), 37),
		{},
	}
	// An empty final chunk still carries a GCM tag — matches the encryptor's
	// fixed-size chunking edge case.

	var enc bytes.Buffer
	binary.Write(&enc, binary.BigEndian, uint32(len(chunks))) //nolint:errcheck
	for _, pt := range chunks {
		nonce := make([]byte, 12)
		if _, err := rand.Read(nonce); err != nil {
			t.Fatal(err)
		}
		ct := aead.Seal(nil, nonce, pt, nil)
		binary.Write(&enc, binary.BigEndian, uint32(len(nonce)+len(ct))) //nolint:errcheck
		enc.Write(nonce)
		enc.Write(ct)
	}

	dir := t.TempDir()
	encPath := filepath.Join(dir, "in.enc")
	outPath := filepath.Join(dir, "out.bin")
	if err := os.WriteFile(encPath, enc.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := DecryptChunkedFile(key, encPath, outPath); err != nil {
		t.Fatalf("decrypt chunked: %v", err)
	}
	got, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	want := bytes.Join(chunks, nil)
	if !bytes.Equal(got, want) {
		t.Fatalf("roundtrip mismatch: got %d bytes, want %d", len(got), len(want))
	}
}
