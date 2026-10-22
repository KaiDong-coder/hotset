// Aggregate streaming read bandwidth of this host, 1..N threads.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NTHREADS 10
#define PER_THREAD (64u << 20)   // 64 MiB per thread, >> 19.25 MiB L3
#define ROUNDS 4

static uint8_t * buf;
static int nthreads;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void * worker(void * arg) {
    long id = (long) arg;
    size_t off = (size_t) id * PER_THREAD;
    uint64_t sum = 0;
    for (int r = 0; r < ROUNDS; ++r) {
        const uint64_t * p = (const uint64_t *) (buf + off);
        for (size_t i = 0; i < PER_THREAD / 8; i += 4) {
            sum += p[i] + p[i+1] + p[i+2] + p[i+3];
        }
    }
    return (void *) sum;
}

int main(void) {
    size_t total = (size_t) NTHREADS * PER_THREAD;
    buf = aligned_alloc(64, total);
    if (!buf) { perror("alloc"); return 1; }
    memset(buf, 1, total);

    for (nthreads = 1; nthreads <= NTHREADS; ++nthreads) {
        pthread_t th[NTHREADS];
        double t0 = now();
        for (long i = 0; i < nthreads; ++i) pthread_create(&th[i], NULL, worker, (void *) i);
        for (int i = 0; i < nthreads; ++i) pthread_join(th[i], NULL);
        double dt = now() - t0;
        double gb = (double) nthreads * PER_THREAD * ROUNDS / 1e9;
        printf("%2d threads: %6.1f GB/s  (%.2f GB/s per thread)\n", nthreads, gb / dt, gb / dt / nthreads);
    }
    free(buf);
    return 0;
}
