#!/bin/sh
# Builds everything the visionOS host app links, for device (xros) and the
# simulator (xrsimulator), into build/s3-visionos:
#
#   scripts/visionos-tree.sh                 both SDKs
#   scripts/visionos-tree.sh xrsimulator     one SDK (xros or xrsimulator)
#
# Per SDK, under build/s3-visionos/<sdk>/:
#   sdl, sdl-inst        SDL3 release-3.4.18, static
#   ffmpeg, ff-inst      FFmpeg n9.0.2 cut to the bink demuxer and decoders,
#                        static, --disable-autodetect so nothing else links
#   spirv-cross          SPIRV-Cross vulkan-sdk-1.4.363.0, static
#   dxvk                 DXVK dd3a05bd: the D3D9 shader front end's
#                        archives and three d3d9 objects, as on the Mac
#   obj                  the lift, the runtime and its shims, recompiled
#   backend              our Metal D3D9 backend, prelinked with the DXVK and
#                        SPIRV-Cross pieces into metal_d3d9.o
#   lib                  libvtmb_game.a (force loaded) and libvtmb_deps.a
# and once for both: porter (the patched tree), include (the slot list and
# host.h) and link.xcconfig, which app/visionos includes.
#
# The runtime and DXVK see the SDL3 headers built here, not Homebrew's,
# and fx.m and hud.m are replaced by the patch series' visionOS stand-ins.
#
# Sources are reused where they already are under build/ and only cloned,
# at their pinned tag, when absent: SDL and FFmpeg from build/s3-visionos/src,
# DXVK and SPIRV-Cross from build/s2-shaders/src,
# which scripts/porter-tree.sh already requires. A pin that does not match
# stops the build. Each middleware build is kept and redone only when its
# configure line changes.
#
# The lift is not repeated. Its compile lines come from the Mac build's
# compile database (build/s2-porter/meson, written by porter-tree.sh), with
# the compiler, target, tree and include paths swapped, and run through
# ninja so a second run only rebuilds what changed.
#
# The backend is prelinked with ld -r and an exported symbol list, so only
# Direct3DCreate9 and the metal_d3d9_ entries stay global; DXVK's and
# SPIRV-Cross's symbols and our internals become local to that object, as
# they are hidden inside the Mac dylib. The backend is compiled with
# -fno-common, or its tentative definitions (the vtables) would stay common
# symbols, which ld -r leaves global.
#
# third_party/ and VtMB/ are never written, nothing reads game data, and the
# output is all under build/, which is gitignored.
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
porter=$repo/third_party/bloodlines-porter
pin=dc88522f5deaa261d0326a2dc2e6b847bbca5dc4
out=$repo/build/s3-visionos
tree=$out/porter
s1=$repo/build/s1-porter/all
mesondb=$repo/build/s2-porter/meson/compile_commands.json
src=$repo/src/metal_d3d9
osver=27.0

sdl_tag=release-3.4.18
sdl_url=https://github.com/libsdl-org/SDL
ff_tag=n9.0.2
ff_url=https://github.com/FFmpeg/FFmpeg
dxvk_rev=dd3a05bde9bae53eecd9b7314761c8c4571cb4f1
sc_tag=vulkan-sdk-1.4.363.0

fail() {
    echo "visionos-tree: $*" >&2
    exit 1
}

sdks=
for a in "$@"; do
    case $a in
    xros | xrsimulator) sdks="$sdks $a" ;;
    *)
        echo "usage: $0 [xros] [xrsimulator]" >&2
        exit 2
        ;;
    esac
done
[ -n "$sdks" ] || sdks="xros xrsimulator"

jobs=$(sysctl -n hw.ncpu)
mkdir -p "$out"
t_all=$(date +%s)

step() {
    printf '%s %s\n' "$(date +%H:%M:%S)" "$*"
}

# Times each piece, so the report and a later run can see where it went.
timed() {
    label=$1
    shift
    t0=$(date +%s)
    "$@"
    echo "$label $(($(date +%s) - t0)) s" | tee -a "$times"
}

# A source tree at an exact tag or commit, found under build/ or cloned at
# that tag. Clean and on the pin, or the build stops.
need_tag() {
    dir=$1
    tag=$2
    git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || return 1
    want=$(git -C "$dir" rev-parse "$tag^{commit}" 2>/dev/null) || return 1
    [ "$(git -C "$dir" rev-parse HEAD)" = "$want" ] || return 1
}

find_src() {
    name=$1
    tag=$2
    url=$3
    shift 3
    for d in "$@"; do
        if [ -d "$d" ]; then
            need_tag "$d" "$tag" || fail "$d is not at $tag"
            [ -z "$(git -C "$d" status --porcelain --untracked-files=no)" ] || fail "$d has local changes"
            echo "$d"
            return
        fi
    done
    d=$out/src/$name
    step "cloning $url at $tag into $d" >&2
    mkdir -p "$out/src"
    git clone --quiet --depth 1 --branch "$tag" "$url" "$d" >&2
    need_tag "$d" "$tag" || fail "$d is not at $tag after cloning"
    echo "$d"
}

sdl_src=$(find_src sdl "$sdl_tag" "$sdl_url" "$out/src/sdl")
ff_src=$(find_src ffmpeg "$ff_tag" "$ff_url" "$out/src/ffmpeg")
dxs=$repo/build/s2-shaders/src/dxvk
scs=$repo/build/s2-shaders/src/spirv-cross
[ -d "$dxs" ] && [ "$(git -C "$dxs" rev-parse HEAD)" = "$dxvk_rev" ] ||
    fail "build/s2-shaders/src/dxvk is missing or not at $dxvk_rev (see AVP-INSTALL.md)"
[ -d "$scs" ] && need_tag "$scs" "$sc_tag" || fail "build/s2-shaders/src/spirv-cross is missing or not at $sc_tag"
for f in "$dxs/subprojects/dxbc-spirv/meson.build" "$dxs/subprojects/libdisplay-info/meson.build"; do
    [ -f "$f" ] || fail "missing $f: DXVK's submodules are not checked out"
done
[ -f "$mesondb" ] || fail "no $mesondb: run scripts/porter-tree.sh first, it writes the compile lines reused here"
[ -f "$s1/modules.txt" ] && [ -f "$s1/link.c" ] || fail "no lift at $s1"
step "sources: SDL $sdl_src, FFmpeg $ff_src, DXVK $dxs, SPIRV-Cross $scs"

# 1. The patched tree, as porter-tree.sh makes it, recreated when the pin
# or the series changes.
[ "$(git -C "$porter" rev-parse HEAD)" = "$pin" ] || fail "third_party/bloodlines-porter is not at $pin"
[ -z "$(git -C "$porter" status --porcelain)" ] || fail "third_party/bloodlines-porter has local changes; it stays pristine"
stamp=$( (echo "$pin"; cat "$repo"/patches/bloodlines-porter/*.patch) | shasum -a 256 | cut -c1-16)
if [ ! -f "$tree/.series-stamp" ] || [ "$(cat "$tree/.series-stamp")" != "$stamp" ]; then
    rm -rf "$tree"
    cp -c -R "$porter" "$tree"
    for p in "$repo"/patches/bloodlines-porter/*.patch; do
        git -C "$tree" apply --check "$p" || fail "$(basename "$p") does not apply"
        git -C "$tree" apply "$p"
    done
    echo "$stamp" >"$tree/.series-stamp"
    step "patched tree made ($stamp)"
else
    step "patched tree up to date ($stamp)"
fi
[ -f "$tree/runtime/sources_visionos.txt" ] && [ -f "$tree/runtime/host.h" ] ||
    fail "the patch series has no visionOS sources list or host.h (patches 0007 and 0010)"
mkdir -p "$out/include"
"$repo/scripts/gen-d3d9-slots.sh" "$tree/shims/d3d9_null.c" "$out/include/mtl_d3d9_slots.h"
cp "$tree/runtime/host.h" "$out/include/host.h"
# The backend's host-facing header (the immersive handoff), for the app's
# bridging header.
cp "$src/immersive.h" "$out/include/immersive.h"

# Runs a configure step only when its command line differs from the last
# one that finished in that directory; the build step after it is always
# run, and is quick when nothing changed.
configured() {
    dir=$1
    shift
    line=$(printf '%s\n' "$@")
    if [ -f "$dir/.configured" ] && [ "$(cat "$dir/.configured")" = "$line" ]; then return 0; fi
    return 1
}

mark_configured() {
    dir=$1
    shift
    printf '%s\n' "$@" >"$dir/.configured"
}

build_sdk() {
    sdk=$1
    case $sdk in
    xros) triple=arm64-apple-xros$osver; subsystem=visionos ;;
    xrsimulator) triple=arm64-apple-xros$osver-simulator; subsystem=visionos-simulator ;;
    esac
    o=$out/$sdk
    mkdir -p "$o"
    times=$o/times.txt
    : >"$times"
    sysroot=$(xcrun --sdk "$sdk" --show-sdk-path)
    cc="xcrun --sdk $sdk clang -target $triple"
    cxx="xcrun --sdk $sdk clang++ -target $triple"
    step "== $sdk ($triple)"

    # 2. SDL3.
    set -- -S "$sdl_src" -B "$o/sdl" -G Ninja -DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT="$sdk" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=$osver -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release \
        -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TEST_LIBRARY=OFF -DSDL_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$o/sdl-inst"
    if ! configured "$o/sdl" "$@"; then
        rm -rf "$o/sdl" "$o/sdl-inst"
        mkdir -p "$o/sdl"
        timed "sdl configure" sh -c 'cmake "$@" >"$0/cfg.log" 2>&1 || { tail -30 "$0/cfg.log"; exit 1; }' "$o/sdl" "$@"
        mark_configured "$o/sdl" "$@"
    fi
    timed "sdl build" sh -c 'ninja -C "$0" >"$0/build.log" 2>&1 && cmake --install "$0" >"$0/install.log" 2>&1 ||
        { tail -30 "$0/build.log"; exit 1; }' "$o/sdl"
    [ -f "$o/sdl-inst/lib/libSDL3.a" ] || fail "no libSDL3.a for $sdk"

    # 3. FFmpeg, bink only.
    set -- --prefix="$o/ff-inst" --disable-everything --disable-autodetect --enable-demuxer=bink \
        --enable-decoder=binkvideo,binkaudio_dct,binkaudio_rdft --enable-protocol=file --disable-programs \
        --disable-doc --enable-static --disable-shared --enable-cross-compile --target-os=darwin --arch=aarch64 \
        --cc="$cc" --sysroot="$sysroot" --extra-cflags="-target $triple" --extra-ldflags="-target $triple"
    if ! configured "$o/ffmpeg" "$@"; then
        rm -rf "$o/ffmpeg" "$o/ff-inst"
        mkdir -p "$o/ffmpeg"
        timed "ffmpeg configure" sh -c 'ff=$1; shift; cd "$0" && "$ff/configure" "$@" >cfg.log 2>&1 ||
            { tail -30 cfg.log; exit 1; }' "$o/ffmpeg" "$ff_src" "$@"
        mark_configured "$o/ffmpeg" "$@"
    fi
    timed "ffmpeg build" sh -c 'make -C "$0" -j"$1" >"$0/make.log" 2>&1 && make -C "$0" install >"$0/inst.log" 2>&1 ||
        { tail -30 "$0/make.log"; exit 1; }' "$o/ffmpeg" "$jobs"
    for l in avformat avcodec avutil; do [ -f "$o/ff-inst/lib/lib$l.a" ] || fail "no lib$l.a for $sdk"; done

    # 4. SPIRV-Cross, the three libraries the backend links.
    set -- -S "$scs" -B "$o/spirv-cross" -G Ninja -DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT="$sdk" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=$osver -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release \
        -DSPIRV_CROSS_CLI=OFF -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_ENABLE_CPP=OFF \
        -DSPIRV_CROSS_ENABLE_C_API=OFF -DSPIRV_CROSS_ENABLE_HLSL=OFF -DSPIRV_CROSS_ENABLE_REFLECT=OFF \
        -DSPIRV_CROSS_ENABLE_UTIL=OFF -DSPIRV_CROSS_SHARED=OFF -DSPIRV_CROSS_STATIC=ON -DSPIRV_CROSS_SKIP_INSTALL=ON
    if ! configured "$o/spirv-cross" "$@"; then
        rm -rf "$o/spirv-cross"
        mkdir -p "$o/spirv-cross"
        timed "spirv-cross configure" sh -c 'cmake "$@" >"$0/cfg.log" 2>&1 || { tail -30 "$0/cfg.log"; exit 1; }' \
            "$o/spirv-cross" "$@"
        mark_configured "$o/spirv-cross" "$@"
    fi
    timed "spirv-cross build" sh -c 'ninja -C "$0" spirv-cross-core spirv-cross-glsl spirv-cross-msl >"$0/build.log" 2>&1 ||
        { tail -30 "$0/build.log"; exit 1; }' "$o/spirv-cross"

    # 5. DXVK's pieces. SDL3 is a headers only dependency of its WSI, found
    # through the pkg-config file SDL installed above.
    cat >"$o/cross.txt" <<EOF
[binaries]
c = ['xcrun', '--sdk', '$sdk', 'clang', '-target', '$triple']
cpp = ['xcrun', '--sdk', '$sdk', 'clang++', '-target', '$triple']
objc = ['xcrun', '--sdk', '$sdk', 'clang', '-target', '$triple']
ar = ['xcrun', '--sdk', '$sdk', 'ar']
strip = ['xcrun', '--sdk', '$sdk', 'strip']
pkg-config = 'pkg-config'

[built-in options]
pkg_config_path = ''

[properties]
pkg_config_libdir = '$o/sdl-inst/lib/pkgconfig'
needs_exe_wrapper = true

[host_machine]
system = 'darwin'
subsystem = '$subsystem'
kernel = 'xnu'
cpu_family = 'aarch64'
cpu = 'arm64'
endian = 'little'
EOF
    set -- --cross-file="$o/cross.txt" -Dbuildtype=debugoptimized -Dnative_sdl3=enabled -Dnative_sdl2=disabled \
        -Dnative_glfw=disabled -Denable_d3d8=false -Denable_d3d10=false -Denable_d3d11=false -Denable_dxgi=false \
        -Dnative_macos=true
    dxb=$o/dxvk
    d3d9p=$dxb/src/d3d9/libdxvk_d3d9.0.dylib.p
    if ! configured "$dxb" "$@" "$(cat "$o/cross.txt")"; then
        rm -rf "$dxb"
        timed "dxvk configure" sh -c 'meson setup "$0" "$@" >"$0.setup.log" 2>&1 ||
            { tail -30 "$0.setup.log"; exit 1; }' "$dxb" "$dxs" "$@"
        mark_configured "$dxb" "$@" "$(cat "$o/cross.txt")"
    fi
    timed "dxvk build" sh -c 'ninja -C "$0" src/dxvk/libdxvk.a src/wsi/libwsi.a src/vulkan/libvkcommon.a \
        src/spirv/libspirv.a src/util/libutil.a subprojects/dxbc-spirv/libdxbc_spv.a \
        subprojects/libdisplay-info/libdisplay-info.a src/d3d9/libdxvk_d3d9.0.dylib.p/d3d9_shader_analysis.cpp.o \
        src/d3d9/libdxvk_d3d9.0.dylib.p/d3d9_constant_copy.cpp.o src/d3d9/libdxvk_d3d9.0.dylib.p/d3d9_util.cpp.o \
        >"$0/build.log" 2>&1 || { tail -30 "$0/build.log"; exit 1; }' "$dxb"

    # 6. The lift, the runtime and its shims, from the Mac compile lines.
    mkdir -p "$o/obj"
    timed "ninja file" python3 -I - "$mesondb" "$repo/build/s2-porter/porter" "$tree" "$o" "$sdk" "$triple" \
        "$o/sdl-inst/include" "$o/ff-inst/include" <<'PY'
# Writes <o>/obj/build.ninja: one edge per compile in the Mac database,
# retargeted. The runtime's list is the tree's sources_visionos.txt, so
# fx.m and hud.m give way to their visionOS stand-ins.
import json, os, shlex, sys
db, mac_tree, tree, o, sdk, triple, sdl_inc, ff_inc = sys.argv[1:]
entries = json.load(open(db))
mac_tree = os.path.realpath(mac_tree)
meson_dir = os.path.dirname(os.path.realpath(db))
vision = [l.strip() for l in open(os.path.join(tree, 'runtime/sources_visionos.txt')) if l.strip()]
mac = [l.strip() for l in open(os.path.join(tree, 'runtime/sources.txt')) if l.strip()]
drop = {os.path.join(tree, f) for f in set(mac) - set(vision)}
add = [os.path.join(tree, f) for f in vision if f not in mac]

def absolute(path):
    p = path if os.path.isabs(path) else os.path.normpath(os.path.join(meson_dir, path))
    if p == mac_tree or p.startswith(mac_tree + '/'):
        p = tree + p[len(mac_tree):]
    return p

def retarget(cmd):
    a = shlex.split(cmd)
    args, src, skip = [], None, 0
    for i, t in enumerate(a):
        if skip:
            skip -= 1
            continue
        if i == 0 or t in ('-MD', '-fdiagnostics-color=always', '-Winvalid-pch'):
            continue
        if t in ('-MQ', '-MF', '-o'):
            skip = 1
            continue
        if t == '-c':
            src = absolute(a[i + 1])
            skip = 1
            continue
        if t.startswith('-I'):
            p = t[2:]
            # Homebrew's SDL and FFmpeg headers are the Mac's; meson's own
            # private folders hold nothing generated for this project.
            if p.startswith('/opt/homebrew') or p == '.' or p.endswith('.p'):
                continue
            t = '-I' + absolute(p)
        elif t.startswith('-ffile-prefix-map='):
            k, v = t[len('-ffile-prefix-map='):].split('=', 1)
            t = '-ffile-prefix-map=%s=%s' % (absolute(k), v)
        elif t == '-include':
            args += [t, absolute(a[i + 1])]
            skip = 1
            continue
        args.append(t)
    return args, src

edges, seen, template = [], set(), None
for e in entries:
    args, src = retarget(e['command'])
    if src in drop:
        continue
    lifted = '/s1-porter/' in src
    if not lifted:
        args += ['-I' + sdl_inc, '-I' + ff_inc]
    if src.endswith('/runtime/up.c'):
        template = args
    edges.append((src, args))
    seen.add(src)
for src in add:
    edges.append((src, template))
names = set()
with open(os.path.join(o, 'obj', 'build.ninja'), 'w') as f:
    f.write('rule cc\n  command = xcrun --sdk %s clang -target %s $flags -MD -MF $out.d -c $in -o $out\n'
            '  depfile = $out.d\n  deps = gcc\n  description = cc $in\n\n' % (sdk, triple))
    objs = []
    for src, args in edges:
        rel = src.split('/s1-porter/all/', 1)[1] if '/s1-porter/all/' in src else os.path.relpath(src, tree)
        name = 'o/' + rel.replace('/', '__').rsplit('.', 1)[0] + '.o'
        assert name not in names, name
        names.add(name)
        objs.append(name)
        f.write('build %s: cc %s\n  flags = %s\n' % (name, src, ' '.join(shlex.quote(x) for x in args)))
    f.write('\nbuild all: phony %s\ndefault all\n' % ' '.join(objs))
with open(os.path.join(o, 'obj', 'objects.txt'), 'w') as f:
    f.write(''.join(os.path.join(o, 'obj', n) + '\n' for n in objs))
print('%d compiles (%d dropped, %d added for visionOS)' % (len(edges), len(drop), len(add)))
PY
    timed "lift and runtime compile" sh -c 'ninja -C "$0" -j"$1" >"$0/ninja.log" 2>&1 ||
        { grep -B2 -A8 "error:" "$0/ninja.log" | head -60; exit 1; }; tail -1 "$0/ninja.log"' "$o/obj" "$jobs"

    # 7. The backend, with the flags porter-tree.sh uses on the Mac plus the
    # SDL headers its visionOS lookup table needs.
    bo=$o/backend
    rm -rf "$bo"
    mkdir -p "$bo/obj"
    cflags="-O2 -g -Wall -Wextra -Werror -fvisibility=hidden -fno-common -I$out/include -I$src -I$o/sdl-inst/include"
    t0=$(date +%s)
    for f in "$src"/*.c; do $cc $cflags -std=gnu11 -c "$f" -o "$bo/obj/$(basename "$f" .c).o"; done
    for f in "$src"/*.m; do $cc $cflags -fobjc-arc -c "$f" -o "$bo/obj/$(basename "$f" .m).o"; done
    $cxx -std=c++17 -O2 -g -fexceptions -fvisibility=hidden \
        -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST -DDXVK_NATIVE_MACOS -DDXVK_WSI_SDL3 \
        -I"$dxb/src/d3d9" -I"$dxs/src/d3d9" -I"$dxb/include" -I"$dxs/include" -I"$dxs/include/vulkan/include" \
        -I"$dxs/include/spirv/include" -I"$dxb/include/native" -I"$dxs/include/native" \
        -I"$dxs/include/native/windows" -I"$dxs/include/native/directx" \
        -I"$dxb/subprojects/dxbc-spirv" -I"$dxs/subprojects/dxbc-spirv" -I"$d3d9p" -I"$scs" -I"$src" \
        -Wno-everything -c "$src/shader.cpp" -o "$bo/obj/shader.o"
    printf '%s\n' _Direct3DCreate9 '_metal_d3d9_*' >"$bo/exports.txt"
    $cxx -r -nostdlib -o "$bo/metal_d3d9.o" "$bo"/obj/*.o \
        "$d3d9p/d3d9_shader_analysis.cpp.o" "$d3d9p/d3d9_constant_copy.cpp.o" "$d3d9p/d3d9_util.cpp.o" \
        "$dxb/src/dxvk/libdxvk.a" "$dxb/src/wsi/libwsi.a" "$dxb/src/vulkan/libvkcommon.a" "$dxb/src/spirv/libspirv.a" \
        "$dxb/src/util/libutil.a" "$dxb/subprojects/dxbc-spirv/libdxbc_spv.a" \
        "$dxb/subprojects/libdisplay-info/libdisplay-info.a" \
        "$o/spirv-cross/libspirv-cross-msl.a" "$o/spirv-cross/libspirv-cross-glsl.a" "$o/spirv-cross/libspirv-cross-core.a" \
        -Wl,-exported_symbols_list,"$bo/exports.txt"
    echo "backend $(($(date +%s) - t0)) s" | tee -a "$times"
    nm -gU "$bo/metal_d3d9.o" | awk '{print $3}' >"$bo/globals.txt"
    if grep -v -e '^_Direct3DCreate9$' -e '^_metal_d3d9_' "$bo/globals.txt" >"$bo/leaked.txt"; then
        fail "the prelinked backend exports more than its entries: $(head -5 "$bo/leaked.txt" | tr '\n' ' ')"
    fi
    grep -q '^_Direct3DCreate9$' "$bo/globals.txt" || fail "the prelinked backend has no Direct3DCreate9"

    # 8. The two archives the app links: the game, force loaded as meson's
    # link_whole loads it on the Mac, and the middleware, loaded on demand.
    mkdir -p "$o/lib"
    cp "$o/obj/objects.txt" "$o/lib/game-objects.txt"
    echo "$bo/metal_d3d9.o" >>"$o/lib/game-objects.txt"
    rm -f "$o/lib/libvtmb_game.a" "$o/lib/libvtmb_deps.a"
    timed "archives" sh -c 'xcrun --sdk "$0" libtool -static -no_warning_for_no_symbols -o "$1/libvtmb_game.a" \
        -filelist "$1/game-objects.txt" && xcrun --sdk "$0" libtool -static -no_warning_for_no_symbols \
        -o "$1/libvtmb_deps.a" "$2/sdl-inst/lib/libSDL3.a" "$2/ff-inst/lib/libavformat.a" \
        "$2/ff-inst/lib/libavcodec.a" "$2/ff-inst/lib/libavutil.a" 2>"$1/libtool.log"' "$sdk" "$o/lib" "$o"

    # The system libraries SDL and FFmpeg name in their pkg-config files,
    # plus what the backend and runtime use and DXVK's C++ runtime.
    python3 -I - "$o/sdl-inst/lib/pkgconfig/sdl3.pc" "$o"/ff-inst/lib/pkgconfig/libav*.pc >"$o/lib/system-libs.txt" <<'PY'
import sys
seen, out = set(), []
for p in sys.argv[1:]:
    for line in open(p):
        if not line.startswith(('Libs:', 'Libs.private:')):
            continue
        toks = line.split(':', 1)[1].split()
        i = 0
        while i < len(toks):
            t = toks[i]
            if t in ('-framework', '-weak_framework') and i + 1 < len(toks):
                t = '-Wl,%s,%s' % (t, toks[i + 1])
                i += 1
            i += 1
            if t.startswith('-L') or t.startswith('-lSDL3') or t.startswith('-lav') or t in ('-pthread', '-lpthread'):
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
for fw in ('Foundation', 'Metal', 'QuartzCore', 'CoreGraphics', 'UIKit'):
    t = '-Wl,-framework,' + fw
    if t not in seen:
        seen.add(t)
        out.append(t)
out.append('-lc++')
print(' '.join(out))
PY
    for f in libvtmb_game.a libvtmb_deps.a; do
        echo "$sdk $f $(du -h "$o/lib/$f" | cut -f1)" | tee -a "$times"
    done
}

for sdk in $sdks; do build_sdk "$sdk"; done

# 9. The settings app/visionos reads, for whichever SDKs have been built.
{
    echo "// Written by scripts/visionos-tree.sh; included by app/visionos/Link.xcconfig."
    echo "VTMB_INCLUDE = $out/include"
    for sdk in xros xrsimulator; do
        [ -f "$out/$sdk/lib/libvtmb_game.a" ] || continue
        echo "VTMB_LDFLAGS[sdk=$sdk*] = -Wl,-force_load,$out/$sdk/lib/libvtmb_game.a $out/$sdk/lib/libvtmb_deps.a $(cat "$out/$sdk/lib/system-libs.txt")"
    done
} >"$out/link.xcconfig"
step "wrote $out/link.xcconfig"
step "done in $(($(date +%s) - t_all)) s"
