/*
 * Batch AEAD encryption for relay workloads.
 *
 * Encrypts multiple packets with different AEAD contexts (keys) in a tight
 * loop. When built with fusion support (PTLS_HAVE_FUSION), dispatches to
 * the AES-NI interleaved batch path for near-linear throughput scaling.
 * Falls back to sequential encryption for non-fusion backends.
 *
 * This file is backend-agnostic at the sequential level: it only calls the
 * public ptls_aead_encrypt() inline, so it works with OpenSSL, fusion,
 * mbedtls, etc.
 */

#include "picotls.h"

#if PTLS_HAVE_FUSION
#include "picotls/fusion.h"

/**
 * Check if all entries in the batch use a fusion AES-GCM backend.
 * If any entry uses a different backend, we must fall back to sequential.
 */
static int batch_all_fusion(ptls_aead_batch_entry_t *entries, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        const ptls_aead_algorithm_t *algo = entries[i].ctx->algo;
        if (algo != &ptls_fusion_aes128gcm && algo != &ptls_fusion_aes256gcm)
            return 0;
    }
    return 1;
}
#endif

static void batch_sequential(ptls_aead_batch_entry_t *entries, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        entries[i].outlen = ptls_aead_encrypt(entries[i].ctx,
            entries[i].output, entries[i].input, entries[i].inlen,
            entries[i].seq, entries[i].aad, entries[i].aadlen);
    }
}

int ptls_aead_encrypt_batch(ptls_aead_batch_entry_t *entries, size_t count)
{
    if (count == 0)
        return 0;

#if PTLS_HAVE_FUSION
    if (batch_all_fusion(entries, count))
        return ptls_fusion_aead_encrypt_batch(entries, count);
#endif

    /* Fallback: sequential encryption */
    batch_sequential(entries, count);
    return 0;
}
