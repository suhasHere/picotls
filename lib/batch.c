/*
 * Batch AEAD encryption for relay workloads.
 *
 * Encrypts multiple packets with different AEAD contexts (keys) in a tight
 * loop. The initial implementation is a simple sequential loop that
 * eliminates per-entry function-call overhead. A future version will
 * interleave AES-NI rounds across entries for true instruction-level
 * parallelism.
 *
 * This file is backend-agnostic: it only calls the public ptls_aead_encrypt()
 * inline, so it works with OpenSSL, fusion, mbedtls, etc.
 */

#include "picotls.h"

int ptls_aead_encrypt_batch(ptls_aead_batch_entry_t *entries, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        entries[i].outlen = ptls_aead_encrypt(entries[i].ctx,
            entries[i].output, entries[i].input, entries[i].inlen,
            entries[i].seq, entries[i].aad, entries[i].aadlen);
    }
    return 0;
}
