package leaderboard

import (
	"crypto/sha1"
	"fmt"
)

// namespaceURL is RFC 4122's NAMESPACE_URL (6ba7b811-9dad-11d1-80b4-00c04fd430c8),
// matching Python's uuid.NAMESPACE_URL.
var namespaceURL = [16]byte{
	0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1,
	0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
}

// UUID5 computes a deterministic RFC 4122 version-5 (SHA-1) UUID, matching
// Python's uuid.uuid5(uuid.NAMESPACE_URL, name). The leaderboard job id is
// uuid5(NAMESPACE_URL, "tanuh:"+job_id) — stable across resubmissions.
func UUID5(name string) string {
	h := sha1.New()
	h.Write(namespaceURL[:])
	h.Write([]byte(name))
	sum := h.Sum(nil)

	var u [16]byte
	copy(u[:], sum[:16])
	u[6] = (u[6] & 0x0f) | 0x50 // version 5
	u[8] = (u[8] & 0x3f) | 0x80 // RFC 4122 variant

	return fmt.Sprintf("%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:16])
}
