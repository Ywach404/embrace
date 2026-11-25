#!/bin/sh
# Builds the Mac tree: the porter with our patch series applied, our Metal
# D3D9 backend, its headless test, and the executable relinked against the
# lift.
#
#   scripts/porter-tree.sh            everything
#   scripts/porter-tree.sh --test     everything, then the headless test
#   scripts/porter-tree.sh --no-link  the patched tree and the backend only
#
# RT_METAL_TSAN=1 builds the backend and its test with ThreadSanitizer
# instead, into build/s2-porter/tsan, and stops there: the game is never
# linked against it. The prebuilt DXVK and SPIRV-Cross archives stay
# uninstrumented.
#
# third_party/bloodlines-porter is never written. The patched copy lives at
# build/s2-porter/porter, an APFS clone (cp -c) of the pinned clone with
# patches/bloodlines-porter/*.patch applied in order by git apply. It is
# recreated only when the pin or a patch changes, because every lifted
# file includes the tree's rt.h and a new copy recompiles all of them.
#
# The backend's shader chain links DXVK's D3D9 front end and SPIRV-Cross
# from their builds under build/s2-shaders (DXVK dd3a05bd with
# dxbc-spirv, SPIRV-Cross vulkan-sdk-1.4.363.0); their sources stay there
# and nothing of theirs is copied into src/. The dylib exports only our
# entry points, so none of their symbols leak into the game process.
#
# The lift is not repeated. The porter's link/meson.build reads every
# generated file from -Dgen_root and only the runtime and shims from its
# own tree, so pointing it at build/s1-porter/all relinks the lift
# with the patched runtime. The output is all under build/s2-porter, which
# is gitignored; nothing here reads VtMB/ or game data.
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
porter=$repo/third_party/bloodlines-porter
pin=dc88522f5deaa261d0326a2dc2e6b847bbca5dc4
s1=$repo/build/s1-porter/all
out=$repo/build/s2-porter
tree=$out/porter
src=$repo/src/metal_d3d9
methods=${RT_METHODS_FILE:-}
shaders=$repo/build/s2-shaders
dxb=$shaders/dxvk-build
dxs=$shaders/src/dxvk
scb=$shaders/spirv-cross-build
scs=$shaders/src/spirv-cross

link=1
test=0
for a in "$@"; do
    case $a in
    --test) test=1 ;;
    --no-link) link=0 ;;
    *)
        echo "usage: $0 [--test] [--no-link]" >&2
        exit 2
        ;;
    esac
done

fail() {
    echo "porter-tree: $*" >&2
    exit 1
}

[ "$(git -C "$porter" rev-parse HEAD)" = "$pin" ] || fail "third_party/bloodlines-porter is not at $pin"
[ -z "$(git -C "$porter" status --porcelain)" ] || fail "third_party/bloodlines-porter has local changes; it stays pristine"
mkdir -p "$out"

# 1. The patched tree, recreated when the pin or the series changes.
stamp=$( (echo "$pin"; cat "$repo"/patches/bloodlines-porter/*.patch) | shasum -a 256 | cut -c1-16)
if [ ! -f "$tree/.series-stamp" ] || [ "$(cat "$tree/.series-stamp")" != "$stamp" ]; then
    rm -rf "$tree"
    cp -c -R "$porter" "$tree"
    for p in "$repo"/patches/bloodlines-porter/*.patch; do
        git -C "$tree" apply --check "$p" || fail "$(basename "$p") does not apply"
        git -C "$tree" apply "$p"
        echo "applied $(basename "$p")"
    done
    echo "$stamp" >"$tree/.series-stamp"
else
    echo "patched tree up to date ($stamp)"
fi

# 2. Slot lists from the patched tree's null device tables.
mkdir -p "$out/include" "$out/lib" "$out/test"
"$repo/scripts/gen-d3d9-slots.sh" "$tree/shims/d3d9_null.c" "$out/include/mtl_d3d9_slots.h"

# 3. The backend.
for f in "$dxb/src/dxvk/libdxvk.a" "$dxb/subprojects/dxbc-spirv/libdxbc_spv.a" "$scb/libspirv-cross-msl.a" \
    "$scs/spirv_msl.hpp"; do
    [ -f "$f" ] || fail "missing $f: the DXVK and SPIRV-Cross builds (build/s2-shaders) are required"
done
san=
libdir=$out/lib
testdir=$out/test
if [ "${RT_METAL_TSAN:-0}" = 1 ]; then
    san=-fsanitize=thread
    libdir=$out/tsan/lib
    testdir=$out/tsan/test
    link=0
    mkdir -p "$libdir" "$testdir"
fi
cflags="-O2 -g -Wall -Wextra -Werror -fvisibility=hidden $san -I$out/include -I$src"
objs=$libdir/obj
rm -rf "$objs"
mkdir -p "$objs"
for f in "$src"/*.c; do
    clang $cflags -std=gnu11 -c "$f" -o "$objs/$(basename "$f" .c).o"
done
for f in "$src"/*.m; do
    clang $cflags -fobjc-arc -c "$f" -o "$objs/$(basename "$f" .m).o"
done
# The chain: DXVK's headers and the flags its objects were built with. DXVK's own warnings are not ours
# to fix, so they are off for this one file.
d3d9p=$dxb/src/d3d9/libdxvk_d3d9.0.dylib.p
clang++ -std=c++17 -O2 -g -fexceptions -fvisibility=hidden $san \
    -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST -DDXVK_NATIVE_MACOS -DDXVK_WSI_SDL3 \
    -I"$dxb/src/d3d9" -I"$dxs/src/d3d9" -I"$dxb/include" -I"$dxs/include" -I"$dxs/include/vulkan/include" \
    -I"$dxs/include/spirv/include" -I"$dxb/include/native" -I"$dxs/include/native" \
    -I"$dxs/include/native/windows" -I"$dxs/include/native/directx" \
    -I"$dxb/subprojects/dxbc-spirv" -I"$dxs/subprojects/dxbc-spirv" -I"$d3d9p" -I"$scs" -I"$src" \
    -Wno-everything -c "$src/shader.cpp" -o "$objs/shader.o"
clang++ -dynamiclib $san -install_name @rpath/libmetal_d3d9.dylib -o "$libdir/libmetal_d3d9.dylib" "$objs"/*.o \
    "$d3d9p/d3d9_shader_analysis.cpp.o" "$d3d9p/d3d9_constant_copy.cpp.o" "$d3d9p/d3d9_util.cpp.o" \
    "$dxb/src/dxvk/libdxvk.a" "$dxb/src/wsi/libwsi.a" "$dxb/src/vulkan/libvkcommon.a" "$dxb/src/spirv/libspirv.a" \
    "$dxb/src/util/libutil.a" "$dxb/subprojects/dxbc-spirv/libdxbc_spv.a" \
    "$dxb/subprojects/libdisplay-info/libdisplay-info.a" \
    "$scb/libspirv-cross-msl.a" "$scb/libspirv-cross-glsl.a" "$scb/libspirv-cross-core.a" \
    -Wl,-dead_strip -Wl,-exported_symbol,_Direct3DCreate9 -Wl,-exported_symbol,_metal_d3d9_set_host \
    -Wl,-exported_symbol,_metal_d3d9_slot_implemented -Wl,-exported_symbol,_metal_d3d9_ff_selfcheck \
    -Wl,-exported_symbol,_metal_d3d9_pair_selfcheck -Wl,-exported_symbol,_metal_d3d9_ff_pair_selfcheck \
    -Wl,-exported_symbol,_metal_d3d9_replay_twice -Wl,-exported_symbol,_metal_d3d9_twin_read \
    -Wl,-exported_symbol,_metal_d3d9_frame_stats -Wl,-exported_symbol,_metal_d3d9_pipeline_stats \
    -Wl,-exported_symbol,_metal_d3d9_pipeline_wait_idle -Wl,-exported_symbol,_metal_d3d9_shutdown \
    -Wl,-exported_symbol,_metal_d3d9_pipeline_fail -Wl,-exported_symbol,_metal_d3d9_conv_level \
    -Wl,-exported_symbol,_metal_d3d9_selfcheck_max_slot \
    -Wl,-exported_symbol,_metal_d3d9_immersive_enable -Wl,-exported_symbol,_metal_d3d9_immersive_enabled \
    -Wl,-exported_symbol,_metal_d3d9_immersive_acquire -Wl,-exported_symbol,_metal_d3d9_immersive_release \
    -Wl,-exported_symbol,_metal_d3d9_immersive_log -Wl,-exported_symbol,_metal_d3d9_immersive_stats \
    -Wl,-exported_symbol,_metal_d3d9_immersive_view_pose \
    -Wl,-exported_symbol,_metal_d3d9_immersive_hand_pose \
    -Wl,-exported_symbol,_metal_d3d9_immersive_read -Wl,-exported_symbol,_metal_d3d9_immersive_frames \
    -Wl,-exported_symbol,_metal_d3d9_immersive_read_layer -Wl,-exported_symbol,'_metal_d3d9_stereo_*' \
    -Wl,-exported_symbol,'_metal_d3d9_space_*' -Wl,-exported_symbol,'_metal_d3d9_fullview_*' \
    -Wl,-exported_symbol,_metal_d3d9_immersive_view_play \
    -framework Foundation -framework Metal -framework QuartzCore -framework CoreGraphics
echo "built $libdir/libmetal_d3d9.dylib"

clang -O1 -g -Wall -Wextra -Werror -std=gnu11 $san -I"$out/include" -I"$src" "$src/test/headless_test.c" \
    "$src/stereo.c" -lz -o "$testdir/metal_d3d9_test"
echo "built $testdir/metal_d3d9_test"
# The immersive handoff's bookkeeping, the space's opening order and the
# screen math, with no Metal.
clang -O1 -g -Wall -Wextra -Werror -std=gnu11 $san -I"$src" "$src/test/handoff_test.c" "$src/handoff.c" "$src/space.c" \
    "$src/stereo.c" -lm -o "$testdir/handoff_test"
echo "built $testdir/handoff_test"

# 4. The executable, relinked against the lift.
if [ "$link" = 1 ]; then
    [ -f "$s1/modules.txt" ] && [ -f "$s1/link.c" ] || fail "no lift at $s1"
    if [ -f "$out/meson/build.ninja" ]; then
        meson setup --reconfigure "$out/meson" "$tree/link" -Dgen_root="$s1" >"$out/meson-setup.log"
    else
        meson setup "$out/meson" "$tree/link" -Dgen_root="$s1" >"$out/meson-setup.log"
    fi
    ninja -C "$out/meson" >"$out/ninja.log" 2>&1 || {
        tail -40 "$out/ninja.log" >&2
        fail "ninja failed, see $out/ninja.log"
    }
    tail -1 "$out/ninja.log"
    echo "built $out/meson/bloodlines-native"
fi

# 5. The launcher, beside the build it launches.
cat >"$out/run-metal.sh" <<'EOF'
#!/bin/sh
# Launches the Mac build with the Metal D3D9 backend (written by
# scripts/porter-tree.sh). Arguments pass through to bloodlines-native, so
# a run names the mode itself: run-metal.sh --run --gfx metal ...
# Display link pacing stays off (RT_PACE=0); in metal mode the
# bridge does not install the pacing hooks at all.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
export RT_METAL_D3D9_LIB=$here/lib/libmetal_d3d9.dylib
export RT_PACE=${RT_PACE:-0}
exec "$here/meson/bloodlines-native" "$@"
EOF
chmod +x "$out/run-metal.sh"
echo "wrote $out/run-metal.sh"

if [ "$test" = 1 ]; then
    "$testdir/handoff_test"
    # The orbs' porthole layout (the app's own arithmetic, no Metal).
    swiftc -O "$repo/app/visionos/Host/OrbLayout.swift" "$repo/app/visionos/test/orb_layout/main.swift" \
        -o "$testdir/orb_layout_test"
    "$testdir/orb_layout_test"
    # The layout edit's math: turns read back, mirrored sides, carrying.
    swiftc -O "$repo/app/visionos/Host/Layout/HUDLayout.swift" "$repo/app/visionos/test/layout/main.swift" \
        -o "$testdir/layout_test"
    "$testdir/layout_test"
    # The hands: the rig, its poses and the renderer offscreen, with the
    # pictures it draws left beside the test to look at.
    swiftc -O "$repo"/app/visionos/Host/Hands/*.swift "$repo/app/visionos/test/hands/main.swift" -o "$testdir/hands_test"
    "$testdir/hands_test" "$testdir/hands" "$repo/assets/hands"
    # The soft target lock: cone, hysteresis, ties, cycling.
    swiftc -O "$repo"/app/visionos/Host/Selection/*.swift "$repo/app/visionos/test/selection/main.swift" \
        -o "$testdir/selection_test"
    "$testdir/selection_test"
    if [ -n "$methods" ]; then "$testdir/metal_d3d9_test" "$libdir/libmetal_d3d9.dylib" "$methods"
    else "$testdir/metal_d3d9_test" "$libdir/libmetal_d3d9.dylib"; fi
fi
