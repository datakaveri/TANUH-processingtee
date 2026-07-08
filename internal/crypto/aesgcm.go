// Package crypto ports the dataset AES-256-GCM decryption formats used by
// the TANUH dataset preparation tooling (tools/prepare_and_upload_datasets.py).
package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
)

// DecryptBlob decrypts a small AES-GCM encrypted blob.
//
// Wire format: [4B big-endian nonce_len][nonce][ciphertext+tag]
func DecryptBlob(key, encData []byte) ([]byte, error) {
	if len(encData) < 4 {
		return nil, fmt.Errorf("crypto: encrypted blob too short: %d bytes", len(encData))
	}
	nonceLen := int(binary.BigEndian.Uint32(encData[:4]))
	if nonceLen <= 0 || 4+nonceLen >= len(encData) {
		return nil, fmt.Errorf("crypto: invalid nonce length %d for blob of %d bytes", nonceLen, len(encData))
	}
	nonce := encData[4 : 4+nonceLen]
	ct := encData[4+nonceLen:]

	aead, err := newAESGCM(key, nonceLen)
	if err != nil {
		return nil, err
	}
	pt, err := aead.Open(nil, nonce, ct, nil)
	if err != nil {
		return nil, fmt.Errorf("crypto: blob authentication failed: %w", err)
	}
	return pt, nil
}

// DecryptChunkedFile decrypts a large AES-GCM encrypted file written in
// chunks (64MB plaintext each at encryption time).
//
// Wire format:
//
//	[4B big-endian: num_chunks]
//	repeated: [4B big-endian: len(nonce+ct)][nonce (12B)][ciphertext+tag]
func DecryptChunkedFile(key []byte, encPath, outPath string) error {
	src, err := os.Open(encPath)
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer dst.Close()

	aead, err := newAESGCM(key, 12)
	if err != nil {
		return err
	}

	var numChunks uint32
	if err := binary.Read(src, binary.BigEndian, &numChunks); err != nil {
		return fmt.Errorf("crypto: read chunk count: %w", err)
	}
	log.Printf("crypto: decrypting %d chunks from %s", numChunks, encPath)

	for i := uint32(0); i < numChunks; i++ {
		var chunkLen uint32
		if err := binary.Read(src, binary.BigEndian, &chunkLen); err != nil {
			return fmt.Errorf("crypto: read chunk %d length: %w", i, err)
		}
		if chunkLen < 12+16 {
			return fmt.Errorf("crypto: chunk %d too short: %d bytes", i, chunkLen)
		}
		chunk := make([]byte, chunkLen)
		if _, err := io.ReadFull(src, chunk); err != nil {
			return fmt.Errorf("crypto: read chunk %d: %w", i, err)
		}
		pt, err := aead.Open(nil, chunk[:12], chunk[12:], nil)
		if err != nil {
			return fmt.Errorf("crypto: chunk %d authentication failed", i)
		}
		if _, err := dst.Write(pt); err != nil {
			return fmt.Errorf("crypto: write chunk %d: %w", i, err)
		}
		if (i+1)%10 == 0 {
			log.Printf("crypto: decrypted chunk %d/%d", i+1, numChunks)
		}
	}
	return dst.Sync()
}

func newAESGCM(key []byte, nonceLen int) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("crypto: aes.NewCipher: %w", err)
	}
	if nonceLen == 12 {
		return cipher.NewGCM(block)
	}
	return cipher.NewGCMWithNonceSize(block, nonceLen)
}
