/* stdatomic.h shim for Apple Clang 6.0 (no C11 atomics header) */
#ifndef _STDATOMIC_H_COMPAT
#define _STDATOMIC_H_COMPAT

typedef enum {
    memory_order_relaxed = __ATOMIC_RELAXED,
    memory_order_consume = __ATOMIC_CONSUME,
    memory_order_acquire = __ATOMIC_ACQUIRE,
    memory_order_release = __ATOMIC_RELEASE,
    memory_order_acq_rel = __ATOMIC_ACQ_REL,
    memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

typedef volatile int atomic_int;
typedef volatile _Bool atomic_bool;
typedef volatile int atomic_flag;

#define ATOMIC_FLAG_INIT 0
#define ATOMIC_VAR_INIT(val) (val)

#define atomic_init(obj, val) (*(obj) = (val))

#define atomic_store(obj, val) __atomic_store_n(obj, val, __ATOMIC_SEQ_CST)
#define atomic_store_explicit(obj, val, order) __atomic_store_n(obj, val, order)

#define atomic_load(obj) __atomic_load_n(obj, __ATOMIC_SEQ_CST)
#define atomic_load_explicit(obj, order) __atomic_load_n(obj, order)

#define atomic_fetch_add(obj, arg) __atomic_fetch_add(obj, arg, __ATOMIC_SEQ_CST)
#define atomic_fetch_add_explicit(obj, arg, order) __atomic_fetch_add(obj, arg, order)

#define atomic_fetch_sub(obj, arg) __atomic_fetch_sub(obj, arg, __ATOMIC_SEQ_CST)
#define atomic_fetch_sub_explicit(obj, arg, order) __atomic_fetch_sub(obj, arg, order)

#define atomic_exchange(obj, val) __atomic_exchange_n(obj, val, __ATOMIC_SEQ_CST)
#define atomic_exchange_explicit(obj, val, order) __atomic_exchange_n(obj, val, order)

#define atomic_compare_exchange_strong(obj, expected, desired) \
    __atomic_compare_exchange_n(obj, expected, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)

#define atomic_compare_exchange_weak(obj, expected, desired) \
    __atomic_compare_exchange_n(obj, expected, desired, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)

static inline _Bool atomic_flag_test_and_set(volatile int *flag) {
    return __atomic_exchange_n(flag, 1, __ATOMIC_SEQ_CST);
}

static inline void atomic_flag_clear(volatile int *flag) {
    __atomic_store_n(flag, 0, __ATOMIC_SEQ_CST);
}

#define atomic_thread_fence(order) __atomic_thread_fence(order)
#define atomic_signal_fence(order) __atomic_signal_fence(order)

#endif /* _STDATOMIC_H_COMPAT */
