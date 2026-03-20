// Stub for __cudaRegisterFatBinary and friends.
// When CUDA runtime (libcudart) is present, these are provided by libcudart.
// When it's absent (weak-linked), these stubs prevent crashes during static init.
// The actual CUDA functions are resolved via the weak dylib when available.

#include <stddef.h>
#include <dlfcn.h>

// These are called by nvcc-generated code at static init to register GPU kernels.
// With weak linking, if libcudart isn't loaded, we need stubs that do nothing.

// Check if the real CUDA runtime is available
static int cuda_available(void) {
    static int checked = 0, available = 0;
    if (!checked) {
        checked = 1;
        available = (dlsym(RTLD_DEFAULT, "cudaGetDeviceCount") != NULL);
    }
    return available;
}

// Forward to real implementation if CUDA is available, otherwise no-op
typedef void** (*register_fat_binary_fn)(void*);
typedef void (*register_fat_binary_end_fn)(void**);
typedef void (*register_function_fn)(void**, const char*, char*, const char*, int, void*, void*, void*, void*, int*);
typedef void (*register_var_fn)(void**, char*, char*, const char*, int, size_t, int, int);
typedef void (*unregister_fat_binary_fn)(void**);

static void* dummy_handle = NULL;

void** __cudaRegisterFatBinary(void *fatCubin) {
    if (cuda_available()) {
        register_fat_binary_fn real = (register_fat_binary_fn)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
        if (real) return real(fatCubin);
    }
    return &dummy_handle;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    if (cuda_available()) {
        register_fat_binary_end_fn real = (register_fat_binary_end_fn)dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");
        if (real) real(fatCubinHandle);
    }
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, void *tid, void *bid,
                            void *bDim, void *gDim, int *wSize) {
    if (cuda_available()) {
        register_function_fn real = (register_function_fn)dlsym(RTLD_NEXT, "__cudaRegisterFunction");
        if (real) real(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    }
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size, int constant, int global) {
    if (cuda_available()) {
        register_var_fn real = (register_var_fn)dlsym(RTLD_NEXT, "__cudaRegisterVar");
        if (real) real(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    }
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    if (cuda_available()) {
        unregister_fat_binary_fn real = (unregister_fat_binary_fn)dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary");
        if (real) real(fatCubinHandle);
    }
}
