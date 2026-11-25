# Installing Embrace on Apple Vision Pro

Embrace is a native visionOS port of Vampire: The Masquerade, Bloodlines by
A. Cervantez, built on the Bloodlines Porter runtime with its own Metal
renderer and full-view stereo.

## Requirements

- A Mac with Apple silicon and Xcode 27.0 or later (visionOS 27 SDK)
- Apple Vision Pro on visionOS 27.0 or later
- An Apple developer account able to sign the extended virtual addressing
  and increased memory limit entitlements the app uses
- A PlayStation VR2 Sense controller pair (recommended, for tracked aiming
  and pointing) or any gamepad visionOS supports, such as a DualSense
- Your own installed copy of Vampire: The Masquerade, Bloodlines for PC
  (see Game files)
- Python 3 (the porter's build and `scripts/visionos-tree.sh` use it; the
  one in Xcode's command line tools is enough)
- Homebrew, with:
  `brew install meson ninja cmake pkg-config glslang sdl3 ffmpeg xcodegen`
- About 10 GB of free disk space for the build

## Game files

You need to supply your own game files, legally obtained from a copy you own.
This repository contains none, and the build refuses to package an app that
carries any.

- The game must be the Steam build, English, patch 1.2: the version
  Bloodlines Porter supports and checks against its file manifest
  (`docs/game-manifest.txt` in that project).
- The build uses Bloodlines Porter's native game build made from your
  install (Build, step 3). The same install is then copied to the
  headset (Install on Apple Vision Pro, step 4).
- On the headset the app looks for the game in its own `VTMB` folder, and
  starts once `Vampire.exe` sits directly inside it (any letter case).
  Everything the game writes (settings, saves, logs, shader caches) goes to
  the app's own storage; your copy is only ever read.

## Build

Start from a checkout of this repository. All output goes under `build/`,
which is gitignored. Run every command from the checkout's root.

1. **Fetch Bloodlines Porter at its pinned commit** into `third_party/`. The
   scripts check the commit and refuse a tree with local changes.

   ```sh
   git clone https://github.com/OlehPetrykUA/bloodlines-porter third_party/bloodlines-porter
   git -C third_party/bloodlines-porter checkout dc88522f5deaa261d0326a2dc2e6b847bbca5dc4
   ```

2. **Fetch and build the shader toolchain.** This clones DXVK from the jm2
   fork at commit `dd3a05bd` (with its dxbc-spirv and libdisplay-info
   submodules) and SPIRV-Cross at tag `vulkan-sdk-1.4.363.0`, and builds
   the pieces the renderer links under `build/s2-shaders/`. A second run
   reuses what is there.

   ```sh
   scripts/shader-deps.sh
   ```

3. **Produce Bloodlines Porter's native game build from your own legally
   obtained, installed copy**, following that project's documentation at
   the pinned commit (v1.0.0, `dc88522`), in a Python environment with the
   porter's pinned tools. Two settings this repository needs: write the
   build's output to `build/s1-porter/all` in this checkout (the porter's
   own default is elsewhere), and keep any helper files the porter
   generates outside `third_party/bloodlines-porter`, since the scripts here
   refuse a porter tree with local changes. The step is done when
   `build/s1-porter/all` holds `modules.txt` and `link.c`. This
   repository's scripts read that folder and nothing else of the game.

4. **Build the Mac tree**: the porter with this repository's patch series
   applied (to a copy; `third_party/` is never written), the Metal renderer
   and its tests. It also writes the compile lines the visionOS build reuses.

   ```sh
   scripts/porter-tree.sh --test
   ```

5. **Build the visionOS libraries** for the device. This fetches SDL
   `release-3.4.18` and FFmpeg `n9.0.2` at their tags under
   `build/s3-visionos/src` on first run.

   ```sh
   scripts/visionos-tree.sh xros
   ```

6. **Set your own bundle identifier.** Bundle identifiers are unique across
   Apple developer teams, so replace `PRODUCT_BUNDLE_IDENTIFIER` in
   `app/visionos/project.yml` with one of yours, for example
   `com.yourname.embrace`.

7. **Build and sign the app** with your team ID (Xcode, Settings, Accounts,
   or the Membership page of your developer account):

   ```sh
   VTMB_TEAM=<your team id> app/visionos/build.sh signed
   ```

   The first signed build registers your bundle identifier and the two
   memory capabilities with Apple, so an Apple account in that team must be
   signed in to Xcode (Settings, Accounts). The two entitlements need a
   paid Apple Developer Program membership.

   The app is at
   `build/s3-visionos/app/DerivedData/Build/Products/Release-xros/XRHost.app`.
   To work in Xcode instead, run `xcodegen generate` in `app/visionos` and
   open `Host.xcodeproj`, with the Host scheme and your team selected.

## Install on Apple Vision Pro

1. Pair the headset with your Mac (on the headset: Settings, General,
   Remote Devices; Developer Mode on), and find its name or identifier with
   `xcrun devicectl list devices`.
2. Install the app:

   ```sh
   xcrun devicectl device install app --device <device> \
       build/s3-visionos/app/DerivedData/Build/Products/Release-xros/XRHost.app
   ```

3. Open Embrace once on the headset. With no game yet, it creates its
   `VTMB` folder and shows where it is.
4. Copy your game into that folder, so `Vampire.exe` sits directly inside
   it:

   ```sh
   xcrun devicectl device copy to --device <device> \
       --domain-type appDataContainer --domain-identifier <your bundle id> \
       --source "<your install folder>" --destination Documents/VTMB
   ```

   The folder also shows in the Files app under On My Apple Vision Pro,
   Embrace, VTMB.

## First launch

Open Embrace from the Home View. It starts in full view, the immersive
6DoF mode, with your head as the camera.

To play in a window instead, launch with `--nofullview`:

```sh
xcrun devicectl device process launch --device <device> <your bundle id> --nofullview
```

In Xcode, add `--nofullview` to the Host scheme's launch arguments.

The gloved hands are off by default. To turn them on, launch with
`RT_HANDS=1` in the environment (from the command line,
`DEVICECTL_CHILD_RT_HANDS=1` before `xcrun devicectl`; in Xcode, an
environment variable on the Host scheme).

## Controls and known issues

- Pair the Sense controllers (or a gamepad) with the headset before
  launching.
- Right trigger fires the weapon in your aiming hand; left trigger uses
  your last discipline.
- In menus, point a controller to move the cursor; Cross or the right
  trigger clicks, Circle goes back.
- Square opens the inventory. Hold Options to switch between first and
  third person; tap it for the game menu.
- Tap Circle to reload; hold it to use doors and talk.
- Click both thumbsticks together (or F8 on a keyboard) to switch between
  full view and the theater screen.
- Hold both thumbsticks in for a second and a half to open the layout
  editor. Grab the orbs, the gloves or the reticle with the trigger, or
  use the gizmo's arrows, rings and scale handle; Cross or Square changes
  the gizmo's mode. Tap Options to save and close.
