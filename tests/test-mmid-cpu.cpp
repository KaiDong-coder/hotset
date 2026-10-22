// Tests for the MXFP4 x Q8_0 fast paths in ggml_compute_forward_mul_mat_id.
//
// The same graph is computed twice in two processes: once with the fast path
// enabled and once with it disabled (GGML_MMID_IL4=0 for the single-token
// path, GGML_MMID_PREFILL=0 for the multi-token path).  The outputs are
// printed on stdout and compared by tests/run_tests.py.  Running the
// two variants in separate processes is deliberate: the toggles are read once
// per process.
//
// Usage: test-mmid-cpu [n_tokens]
//   n_tokens = 1 exercises the interleaved four-expert decode kernel
//   n_tokens = 8 exercises the r2t4 prefill kernel

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int N_EMBD   = 256;
static const int N_EXPERT = 8;
static const int N_USED   = 4;

// deterministic MXFP4 payload: e in {120..122} keeps the values finite
static void fill_weights(struct ggml_tensor * w) {
    uint8_t * p = (uint8_t *) w->data;
    const int n_blocks = (int) (ggml_nelements(w) / 32);
    for (int b = 0; b < n_blocks; ++b) {
        uint8_t * blk = p + (size_t) b * 17;
        blk[0] = (uint8_t) (120 + (b % 3));
        for (int i = 0; i < 16; ++i) {
            blk[1 + i] = (uint8_t) ((b*37 + i*17 + 11) & 0xFF);
        }
    }
}

int main(int argc, char ** argv) {
    const int n_tok = argc > 1 ? atoi(argv[1]) : 1;

    struct ggml_init_params ip = {
        /*.mem_size   =*/ 32*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    // as [cols, rows, n_expert], b [cols, 1, n_tokens], ids [n_used, n_tokens]
    struct ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, N_EMBD, N_EMBD, N_EXPERT);
    struct ggml_tensor * act = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,   N_EMBD, 1, n_tok);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,   N_USED, n_tok);

    struct ggml_tensor * out = ggml_mul_mat_id(ctx, w, act, ids);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    fill_weights(w);

    float * a = (float *) act->data;
    for (int i = 0; i < N_EMBD*n_tok; ++i) {
        a[i] = sinf(0.017f * (float) i) * 1.5f + 0.25f*cosf(0.031f*(float)i);
    }
    int32_t * id = (int32_t *) ids->data;
    for (int t = 0; t < n_tok; ++t) {
        for (int s = 0; s < N_USED; ++s) {
            id[t*N_USED + s] = (s*3 + t*5 + 1) % N_EXPERT;
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    const float * o = (const float *) out->data;
    const int64_t n = ggml_nelements(out);
    printf("n_tokens %d n_out %lld w[%lld,%lld,%lld] act[%lld,%lld,%lld] ids[%lld,%lld] out[%lld,%lld,%lld]\n",
           n_tok, (long long) n,
           (long long) w->ne[0], (long long) w->ne[1], (long long) w->ne[2],
           (long long) act->ne[0], (long long) act->ne[1], (long long) act->ne[2],
           (long long) ids->ne[0], (long long) ids->ne[1],
           (long long) out->ne[0], (long long) out->ne[1], (long long) out->ne[2]);
    for (int64_t i = 0; i < n; ++i) {
        printf("%.9g\n", o[i]);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
