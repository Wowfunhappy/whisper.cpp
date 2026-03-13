#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

SCRIPT_DIR=scripts/manual-build
BUILD=build-manual
CC=clang
CXX=clang++

mkdir -p "${BUILD}"

INCLUDES="-I./${SCRIPT_DIR} -I./include -I./examples -I./src -I./ggml/include -I./ggml/src -I./ggml/src/ggml-cpu -I./ggml/src/ggml-cpu/arch -I./ggml/src/ggml-cpu/llamafile -I./ggml/src/ggml-blas"
DEFS="-DNDEBUG -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DGGML_USE_CPU -DGGML_USE_BLAS -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_BLAS_USE_ACCELERATE"
PREINCLUDE="-include ${SCRIPT_DIR}/compat.h -include ${SCRIPT_DIR}/config.h"

# SIMD flags for i7-4790K (Haswell): AVX2, FMA, BMI1/2, SSE4.2
# Note: -mf16c omitted because Apple Clang 6.0 lacks F16C intrinsic headers
SIMD="-mavx2 -mfma -mbmi -mbmi2 -msse4.2 -mpopcnt"

CFLAGS_BASE="-std=c11 -O3 $SIMD $DEFS $PREINCLUDE $INCLUDES"
CXXFLAGS_BASE="-std=c++1y -O3 $SIMD $DEFS $PREINCLUDE $INCLUDES"

OBJS=""

compile_c() {
    local src=$1
    local obj="${BUILD}/$(echo "$src" | sed 's|/|_|g' | sed 's|\.c$|_c.o|')"
    echo "  CC  $src"
    $CC $CFLAGS_BASE -c "$src" -o "$obj"
    OBJS="$OBJS $obj"
}

compile_cpp() {
    local src=$1
    local obj="${BUILD}/$(echo "$src" | sed 's|/|_|g' | sed 's|\.cpp$|_cpp.o|')"
    echo "  CXX $src"
    $CXX $CXXFLAGS_BASE -c "$src" -o "$obj"
    OBJS="$OBJS $obj"
}

echo "=== Building whisper-cli ==="
echo ""

echo "--- GGML core (C) ---"
compile_c ggml/src/ggml.c
compile_c ggml/src/ggml-alloc.c
compile_c ggml/src/ggml-quants.c

echo "--- GGML core (C++) ---"
compile_cpp ggml/src/ggml.cpp
compile_cpp ggml/src/ggml-backend.cpp
compile_cpp ggml/src/ggml-backend-reg.cpp
compile_cpp ggml/src/ggml-backend-dl.cpp
compile_cpp ggml/src/ggml-opt.cpp
compile_cpp ggml/src/ggml-threading.cpp
compile_cpp ggml/src/gguf.cpp

echo "--- GGML CPU backend ---"
compile_c ggml/src/ggml-cpu/ggml-cpu.c
compile_cpp ggml/src/ggml-cpu/ggml-cpu.cpp
compile_cpp ggml/src/ggml-cpu/ops.cpp
compile_cpp ggml/src/ggml-cpu/vec.cpp
compile_cpp ggml/src/ggml-cpu/binary-ops.cpp
compile_cpp ggml/src/ggml-cpu/unary-ops.cpp
compile_c ggml/src/ggml-cpu/quants.c
compile_cpp ggml/src/ggml-cpu/repack.cpp
compile_cpp ggml/src/ggml-cpu/traits.cpp
compile_cpp ggml/src/ggml-cpu/hbm.cpp
compile_cpp ggml/src/ggml-cpu/llamafile/sgemm.cpp

echo "--- GGML CPU arch (x86) ---"
compile_c ggml/src/ggml-cpu/arch/x86/quants.c
compile_cpp ggml/src/ggml-cpu/arch/x86/repack.cpp
compile_cpp ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp

echo "--- GGML BLAS/Accelerate backend ---"
compile_cpp ggml/src/ggml-blas/ggml-blas.cpp

echo "--- Whisper core ---"
compile_cpp src/whisper.cpp

echo "--- Whisper common ---"
compile_cpp examples/common.cpp
compile_cpp examples/common-ggml.cpp
compile_cpp examples/common-whisper.cpp
compile_cpp examples/grammar-parser.cpp

echo "--- Whisper CLI ---"
compile_cpp examples/cli/cli.cpp

echo ""
echo "--- Linking ---"
echo "  LD  ${BUILD}/whisper-cli"
$CXX -o "${BUILD}/whisper-cli" $OBJS \
    /usr/local/lib/libMacportsLegacySupport.a \
    -framework Accelerate \
    -lpthread -ldl

echo ""
echo "=== Build complete: ${BUILD}/whisper-cli ==="
ls -lh "${BUILD}/whisper-cli"
