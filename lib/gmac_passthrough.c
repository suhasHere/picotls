/*
 * GMAC Passthrough AEAD for QUIC-RT
 *
 * Provides authentication without encryption for pre-encrypted payloads.
 * Used by relay fan-out where the payload is group-encrypted once and
 * the per-connection AEAD only needs to authenticate (not re-encrypt).
 *
 * On encrypt: memcpy payload to output + compute GMAC tag
 * On decrypt: verify GMAC tag + memcpy payload to output
 *
 * The GMAC tag is computed using the same AES key's GHASH subkey,
 * making it verifiable by both relay and subscriber.
 */

#include <string.h>
#include <assert.h>
#include <openssl/evp.h>
#include "picotls.h"
#include "picotls/openssl.h"

struct gmac_context_t {
    ptls_aead_context_t super;
    EVP_CIPHER_CTX *evp_ctx;
    uint8_t static_iv[PTLS_MAX_IV_SIZE];
};

static void gmac_dispose_crypto(ptls_aead_context_t *_ctx)
{
    struct gmac_context_t *ctx = (struct gmac_context_t *)_ctx;
    if (ctx->evp_ctx != NULL)
        EVP_CIPHER_CTX_free(ctx->evp_ctx);
}

static void gmac_get_iv(ptls_aead_context_t *_ctx, void *iv)
{
    struct gmac_context_t *ctx = (struct gmac_context_t *)_ctx;
    memcpy(iv, ctx->static_iv, ctx->super.algo->iv_size);
}

static void gmac_set_iv(ptls_aead_context_t *_ctx, const void *iv)
{
    struct gmac_context_t *ctx = (struct gmac_context_t *)_ctx;
    memcpy(ctx->static_iv, iv, ctx->super.algo->iv_size);
}

/*
 * GMAC encrypt: memcpy payload + compute authentication tag.
 * No AES-CTR encryption — payload passes through unchanged.
 * Tag is computed as GMAC(K, nonce, AAD || payload).
 */
static void gmac_do_encrypt(ptls_aead_context_t *_ctx, void *output,
    const void *input, size_t inlen, uint64_t seq,
    const void *aad, size_t aadlen, ptls_aead_supplementary_encryption_t *supp)
{
    static int gmac_enc_count = 0;
    if (++gmac_enc_count <= 5) {
        fprintf(stderr, "[GMAC] encrypt #%d: inlen=%zu aadlen=%zu seq=%llu\n",
                gmac_enc_count, inlen, aadlen, (unsigned long long)seq);
        fflush(stderr);
    }
    struct gmac_context_t *ctx = (struct gmac_context_t *)_ctx;
    uint8_t iv[PTLS_MAX_IV_SIZE];
    int blocklen, ret;
    size_t tag_size = ctx->super.algo->tag_size;

    /* Build nonce from static IV XOR sequence number */
    ptls_aead__build_iv(ctx->super.algo, iv, ctx->static_iv, seq);

    /* Initialize with the key and nonce */
    ret = EVP_EncryptInit_ex(ctx->evp_ctx, NULL, NULL, NULL, iv);
    assert(ret);

    /* Feed QUIC header as AAD */
    if (aadlen > 0) {
        ret = EVP_EncryptUpdate(ctx->evp_ctx, NULL, &blocklen, aad, (int)aadlen);
        assert(ret);
    }

    /* Feed payload as AAD (not as plaintext — this is GMAC, not GCM) */
    if (inlen > 0) {
        ret = EVP_EncryptUpdate(ctx->evp_ctx, NULL, &blocklen, input, (int)inlen);
        assert(ret);
    }

    /* Finalize and get the GMAC tag */
    ret = EVP_EncryptFinal_ex(ctx->evp_ctx, (uint8_t *)output + inlen, &blocklen);
    assert(ret);
    ret = EVP_CIPHER_CTX_ctrl(ctx->evp_ctx, EVP_CTRL_GCM_GET_TAG, (int)tag_size,
                               (uint8_t *)output + inlen);
    assert(ret);

    /* Copy payload to output (passthrough — no encryption) */
    memcpy(output, input, inlen);
}

/*
 * GMAC decrypt: verify authentication tag + memcpy payload.
 * Returns payload length on success, SIZE_MAX on tag verification failure.
 */
static size_t gmac_do_decrypt(ptls_aead_context_t *_ctx, void *output,
    const void *input, size_t inlen, uint64_t seq,
    const void *aad, size_t aadlen)
{
    static int gmac_dec_count = 0;
    if (++gmac_dec_count <= 5) {
        fprintf(stderr, "[GMAC] decrypt #%d: inlen=%zu aadlen=%zu seq=%llu\n",
                gmac_dec_count, inlen, aadlen, (unsigned long long)seq);
        fflush(stderr);
    }
    struct gmac_context_t *ctx = (struct gmac_context_t *)_ctx;
    uint8_t iv[PTLS_MAX_IV_SIZE];
    int blocklen, ret;
    size_t tag_size = ctx->super.algo->tag_size;

    if (inlen < tag_size)
        return SIZE_MAX;

    size_t payload_len = inlen - tag_size;

    ptls_aead__build_iv(ctx->super.algo, iv, ctx->static_iv, seq);

    /* Initialize decrypt context */
    ret = EVP_DecryptInit_ex(ctx->evp_ctx, NULL, NULL, NULL, iv);
    assert(ret);

    /* Feed QUIC header as AAD */
    if (aadlen > 0) {
        ret = EVP_DecryptUpdate(ctx->evp_ctx, NULL, &blocklen, aad, (int)aadlen);
        assert(ret);
    }

    /* Feed payload as AAD (same as encrypt — GMAC treats payload as AAD) */
    if (payload_len > 0) {
        ret = EVP_DecryptUpdate(ctx->evp_ctx, NULL, &blocklen, input, (int)payload_len);
        assert(ret);
    }

    /* Set the tag for verification */
    if (!EVP_CIPHER_CTX_ctrl(ctx->evp_ctx, EVP_CTRL_GCM_SET_TAG, (int)tag_size,
                              (void *)((const uint8_t *)input + payload_len)))
        return SIZE_MAX;

    /* Verify tag */
    if (!EVP_DecryptFinal_ex(ctx->evp_ctx, (uint8_t *)output + payload_len, &blocklen)) {
        if (gmac_dec_count <= 5) {
            fprintf(stderr, "[GMAC] decrypt FAILED tag verify! inlen=%zu\n", inlen);
            fflush(stderr);
        }
        return SIZE_MAX;
    }

    if (gmac_dec_count <= 5) {
        fprintf(stderr, "[GMAC] decrypt OK! payload=%zu\n", payload_len);
        fflush(stderr);
    }

    /* Copy payload to output (passthrough) */
    memcpy(output, input, payload_len);

    return payload_len;
}

/*
 * Create a GMAC passthrough AEAD context from an existing AES-GCM context.
 * Uses the same key and IV but replaces encrypt/decrypt with GMAC passthrough.
 */
int ptls_gmac_setup_crypto(ptls_aead_context_t *_ctx, int is_enc, const void *key,
                            const void *iv)
{
    struct gmac_context_t *ctx = (struct gmac_context_t *)_ctx;
    int ret;

    ctx->evp_ctx = EVP_CIPHER_CTX_new();
    if (ctx->evp_ctx == NULL)
        return PTLS_ERROR_NO_MEMORY;

    if (is_enc) {
        ret = EVP_EncryptInit_ex(ctx->evp_ctx, EVP_aes_128_gcm(), NULL, key, NULL);
    } else {
        ret = EVP_DecryptInit_ex(ctx->evp_ctx, EVP_aes_128_gcm(), NULL, key, NULL);
    }
    if (!ret) {
        EVP_CIPHER_CTX_free(ctx->evp_ctx);
        ctx->evp_ctx = NULL;
        return PTLS_ERROR_LIBRARY;
    }

    /* Set IV length */
    ret = EVP_CIPHER_CTX_ctrl(ctx->evp_ctx, EVP_CTRL_GCM_SET_IVLEN,
                               (int)_ctx->algo->iv_size, NULL);
    assert(ret);

    ctx->super.dispose_crypto = gmac_dispose_crypto;
    ctx->super.do_get_iv = gmac_get_iv;
    ctx->super.do_set_iv = gmac_set_iv;
    ctx->super.do_encrypt_init = NULL;
    ctx->super.do_encrypt_update = NULL;
    ctx->super.do_encrypt_final = NULL;

    if (is_enc) {
        ctx->super.do_encrypt = gmac_do_encrypt;
        ctx->super.do_encrypt_v = NULL;
        ctx->super.do_decrypt = NULL;
    } else {
        ctx->super.do_encrypt = NULL;
        ctx->super.do_encrypt_v = NULL;
        ctx->super.do_decrypt = gmac_do_decrypt;
    }

    memcpy(ctx->static_iv, iv, _ctx->algo->iv_size);

    return 0;
}
