#!/bin/sh
# Refuses a shareable build of the app that carries any game data.
#
#   refuse-game-data.sh [--delete PATH]... ARTIFACT...
#       Checks each ARTIFACT (an .app, an .xcarchive, or an .ipa). If any
#       holds game data, names every offending entry, deletes each ARTIFACT
#       and each --delete PATH, so a stale copy cannot be shared in its
#       place, and exits 1.
#
#   refuse-game-data.sh --phase
#       The Xcode build phase. On an archive (ACTION=install) it checks the
#       bundle being built and, on a find, deletes it and the archive and
#       fails the build. On any other build, a simulator run included, it
#       only warns: running locally never needs the shareable artifact.
#
# Game data is anything with a game file extension or a folder named like
# the game's own. The app is built to carry none; the player copies their
# own install into the app's Documents folder at run time. This check is
# what makes "none" hold when someone adds a resource by mistake.
set -eu
# The word lists below are split unquoted; nothing in them may glob.
set -f

# Extensions from the game's packs, maps, models, materials, saves and
# binaries, matched without regard to case.
exts='vpk bsp mdl vtf vmt vvd vtx phy sav exe dll'
# Top level folders of an install.
dirs='vampire bin'

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Prints one line per offending entry of a list of relative paths on stdin.
offending() {
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        lp=$(lower "$p")
        why=
        for e in $exts; do
            case $lp in *."$e" | *."$e"/) why="a .$e file" ;; esac
        done
        for d in $dirs; do
            case /$lp/ in */"$d"/*) why=${why:-"inside a $d/ folder"} ;; esac
        done
        [ -z "$why" ] || printf '%s (%s)\n' "$p" "$why"
    done
}

# Lists an artifact's entries, relative to its root.
entries() {
    case $1 in
    *.ipa | *.zip) unzip -Z1 "$1" ;;
    *) (cd "$1" && find . -mindepth 1 | sed 's|^\./||') ;;
    esac
}

check() {
    [ -e "$1" ] || return 0
    found=$(entries "$1" | offending)
    [ -z "$found" ] && return 0
    printf '%s\n' "$found" >"${TMPDIR:-/tmp}/refuse-game-data.$$"
    return 1
}

report() {
    level=$1
    artifact=$2
    echo "$level: $artifact holds game data, which must never ship in a build others can download:"
    sed 's/^/    /' "${TMPDIR:-/tmp}/refuse-game-data.$$"
    rm -f "${TMPDIR:-/tmp}/refuse-game-data.$$"
}

if [ "${1:-}" = --phase ]; then
    bundle=${CODESIGNING_FOLDER_PATH:-${TARGET_BUILD_DIR:?}/${WRAPPER_NAME:?}}
    if check "$bundle"; then
        echo "refuse-game-data: $bundle holds no game data"
        exit 0
    fi
    if [ "${ACTION:-build}" != install ]; then
        report warning "$bundle"
        echo "warning: a local build may run like this, but an archive of it will be refused"
        exit 0
    fi
    report error "$bundle"
    echo "error: archive refused; deleting $bundle${ARCHIVE_PATH:+ and $ARCHIVE_PATH}"
    rm -rf "$bundle"
    [ -z "${ARCHIVE_PATH:-}" ] || rm -rf "$ARCHIVE_PATH"
    exit 1
fi

stale=
while [ $# -gt 0 ]; do
    case $1 in
    --delete)
        [ $# -ge 2 ] || { echo "usage: $0 [--delete PATH]... ARTIFACT..." >&2; exit 2; }
        stale="$stale
$2"
        shift 2
        ;;
    *) break ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: $0 [--delete PATH]... ARTIFACT... | --phase" >&2; exit 2; }

bad=0
for a in "$@"; do
    if check "$a"; then
        echo "refuse-game-data: $a holds no game data"
    else
        report error "$a"
        bad=1
    fi
done
[ "$bad" = 0 ] && exit 0
printf '%s\n' "$stale" "$@" | while IFS= read -r p; do
    [ -n "$p" ] || continue
    [ -e "$p" ] || continue
    echo "refuse-game-data: deleting $p"
    rm -rf "$p"
done
echo "error: refused; nothing shareable was left behind"
exit 1
