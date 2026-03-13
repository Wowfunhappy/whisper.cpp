#ifndef COMPAT_LEGACY_H
#define COMPAT_LEGACY_H

/* clock_gettime support for macOS < 10.12 via MacportsLegacySupport */
#include <time.h>
#include <sys/time.h>

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME  0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 6
#endif
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#ifndef CLOCK_PROCESS_CPUTIME_ID
#define CLOCK_PROCESS_CPUTIME_ID 12
#endif
#ifndef CLOCK_THREAD_CPUTIME_ID
#define CLOCK_THREAD_CPUTIME_ID 16
#endif

#ifndef __CLOCKID_T_DEFINED
typedef int clockid_t;
#define __CLOCKID_T_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif
int clock_gettime(clockid_t clk_id, struct timespec *tp);
int clock_getres(clockid_t clk_id, struct timespec *res);
#ifdef __cplusplus
}
#endif

#endif /* COMPAT_LEGACY_H */
