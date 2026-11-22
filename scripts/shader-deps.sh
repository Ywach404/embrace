#!/bin/sh
# Fetches and builds the shader toolchain the Metal renderer translates
# the game's shaders with, for the Mac, under build/s2-shaders:
#   DXVK (the jm2 fork, its D3D9 shader front end, with dxbc-spirv and
#   libdisplay-info as submodules) and SPIRV-Cross (SPIR-V to Metal).
# scripts/porter-tree.sh needs both. Pinned, so every checkout builds the
# same thing; a second run reuses what is there.
#
#   scripts/shader-deps.sh
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/build/s2-shaders
src=$out/src
dxvk_url=https://github.com/jm2/dxvk
dxvk_commit=dd3a05bde9bae53eecd9b7314761c8c4571cb4f1
spvc_url=https://github.com/KhronosGroup/SPIRV-Cross
spvc_tag=vulkan-sdk-1.4.363.0

for tool in git meson ninja cmake; do
    command -v "$tool" >/dev/null || { echo "$tool not found (see AVP-INSTALL.md, Requirements)" >&2; exit 1; }
done
mkdir -p "$src"

if [ ! -d "$src/dxvk/.git" ]; then
    git clone "$dxvk_url" "$src/dxvk"
fi
git -C "$src/dxvk" fetch -q origin "$dxvk_commit" 2>/dev/null || :
git -C "$src/dxvk" checkout -q "$dxvk_commit"
git -C "$src/dxvk" submodule update --init --recursive

if [ ! -d "$src/spirv-cross/.git" ]; then
    git clone --depth 1 --branch "$spvc_tag" "$spvc_url" "$src/spirv-cross"
fi

# Only the pieces the renderer links: the core, its WSI and utility
# libraries, and the three D3D9 shader translation units.
if [ ! -f "$out/dxvk-build/build.ninja" ]; then
    meson setup "$out/dxvk-build" "$src/dxvk" \
        -Dbuildtype=debugoptimized -Dnative_macos=true -Dnative_sdl3=enabled \
        -Denable_d3d8=false -Denable_d3d10=false -Denable_d3d11=false -Denable_dxgi=false
fi
ninja -C "$out/dxvk-build" src/dxvk/libdxvk.a src/wsi/libwsi.a \
    src/vulkan/libvkcommon.a src/spirv/libspirv.a src/util/libutil.a \
    subprojects/dxbc-spirv/libdxbc_spv.a subprojects/libdisplay-info/libdisplay-info.a \
    src/d3d9/libdxvk_d3d9.0.dylib.p/d3d9_shader_analysis.cpp.o \
    src/d3d9/libdxvk_d3d9.0.dylib.p/d3d9_constant_copy.cpp.o \
    src/d3d9/libdxvk_d3d9.0.dylib.p/d3d9_util.cpp.o

if [ ! -f "$out/spirv-cross-build/build.ninja" ]; then
    cmake -S "$src/spirv-cross" -B "$out/spirv-cross-build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DSPIRV_CROSS_ENABLE_TESTS=OFF \
        -DSPIRV_CROSS_SHARED=OFF -DSPIRV_CROSS_STATIC=ON
fi
ninja -C "$out/spirv-cross-build" spirv-cross-core spirv-cross-glsl spirv-cross-msl

echo "shader toolchain ready under build/s2-shaders"
