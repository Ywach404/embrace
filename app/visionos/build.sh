#!/bin/sh
# Builds the visionOS host app from the command line. Nothing here boots a
# simulator, installs, launches or opens a window.
#
#   app/visionos/build.sh sim      Release build for any visionOS simulator
#   app/visionos/build.sh device   Release build for device, unsigned
#                                  (CODE_SIGNING_ALLOWED=NO), to prove the link
#   app/visionos/build.sh signed   Release build for device, signed for
#                                  installing on your own headset; needs
#                                  VTMB_TEAM set to the signing team ID,
#                                  which is kept out of the repository
#   app/visionos/build.sh ipa      archive and export a signed .ipa, the one
#                                  artifact that is ever shared; needs a
#                                  signing identity and team set in Xcode
#
# The game and its runtime come from scripts/visionos-tree.sh, run first for
# the SDK being built. Output, DerivedData included, is under
# build/s3-visionos/app. The app bundle carries no game data in any mode; an
# archive or export that does is refused and deleted (scripts/refuse-game-data.sh).
set -eu

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=$repo/build/s3-visionos/app
dd=$out/DerivedData
mode=${1:-}

fail() {
    echo "build.sh: $*" >&2
    exit 1
}

case $mode in
sim) sdk=xrsimulator; dest='generic/platform=visionOS Simulator' ;;
device | signed | ipa) sdk=xros; dest='generic/platform=visionOS' ;;
*)
    echo "usage: $0 sim|device|signed|ipa" >&2
    exit 2
    ;;
esac

grep -q "^VTMB_LDFLAGS\[sdk=$sdk\*\]" "$repo/build/s3-visionos/link.xcconfig" 2>/dev/null ||
    fail "nothing built for $sdk yet: run scripts/visionos-tree.sh $sdk"
mkdir -p "$out"
(cd "$here" && xcodegen generate --quiet)

common="-project $here/Host.xcodeproj -scheme Host -configuration Release -derivedDataPath $dd"
case $mode in
sim | device | signed)
    extra=
    [ "$mode" = device ] && extra=CODE_SIGNING_ALLOWED=NO
    if [ "$mode" = signed ]; then
        [ -n "${VTMB_TEAM:-}" ] || fail "set VTMB_TEAM to the signing team ID"
        # Automatic signing registers the bundle ID and the entitlements'
        # capabilities with the team on first use.
        extra="DEVELOPMENT_TEAM=$VTMB_TEAM CODE_SIGN_STYLE=Automatic -allowProvisioningUpdates"
    fi
    # shellcheck disable=SC2086
    xcodebuild $common -destination "$dest" $extra build >"$out/$mode.log" 2>&1 || {
        grep -E 'error:|warning: .*refuse' "$out/$mode.log" | head -40 >&2
        fail "xcodebuild failed, see $out/$mode.log"
    }
    case $sdk in xrsimulator) p=Release-xrsimulator ;; xros) p=Release-xros ;; esac
    app=$dd/Build/Products/$p/XRHost.app
    [ -d "$app" ] || fail "no $app"
    echo "built $app ($(du -sh "$app" | cut -f1))"
    ;;
ipa)
    archive=$out/Host.xcarchive
    export=$out/export
    # Neither an older archive nor an older .ipa may outlive a refused one.
    rm -rf "$archive" "$export"
    # shellcheck disable=SC2086
    xcodebuild $common -destination "$dest" -archivePath "$archive" archive >"$out/archive.log" 2>&1 || {
        grep -E 'error:|^    ' "$out/archive.log" | head -40 >&2
        rm -rf "$archive" "$export"
        fail "archive failed or was refused, see $out/archive.log"
    }
    "$here/scripts/refuse-game-data.sh" --delete "$export" "$archive"
    cat >"$out/export-options.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key>
	<string>debugging</string>
</dict>
</plist>
EOF
    xcodebuild -exportArchive -archivePath "$archive" -exportPath "$export" \
        -exportOptionsPlist "$out/export-options.plist" >"$out/export.log" 2>&1 || {
        rm -rf "$export"
        fail "export failed, see $out/export.log"
    }
    for ipa in "$export"/*.ipa; do
        "$here/scripts/refuse-game-data.sh" --delete "$archive" "$ipa"
        echo "exported $ipa ($(du -h "$ipa" | cut -f1))"
    done
    ;;
esac
