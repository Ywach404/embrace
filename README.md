<img src="docs/media/icon.png" alt="Embrace app icon" width="200">

# Embrace

**The night of Los Angeles, all around you on Apple Vision Pro.**

<img src="docs/media/embrace-downtown.gif" alt="Downtown Los Angeles in full view on Apple Vision Pro: a hand-aimed rifle, then a conversation" width="480">

<img src="docs/media/hand-aimed-rifle.jpg" alt="First person, the rifle held and aimed with the controller" width="260"> <img src="docs/media/third-person-street.jpg" alt="Third person on a downtown street, the world held still around the player" width="260"> <img src="docs/media/side-gauges.jpg" alt="The health and blood gauges riding on the controllers" width="260">

## What it is

Embrace is a native visionOS port of Vampire: The Masquerade, Bloodlines
(Troika Games, 2004), built by A. Cervantez as a portfolio piece. The game
runs natively on the headset's Apple silicon and draws through a Metal
renderer written for this port, in true stereo, with your head as the
camera.

## Features

- **Full-view 6DoF stereo.** Each eye gets its own render of the world
  from its own position, filling the whole field of view. Your head is the
  camera: look around, lean, and the city moves as it should.
- **Native Metal renderer.** The game's Direct3D 9 calls go straight to a
  Metal backend made for this port, with no Vulkan layer underneath. Its
  shaders are translated to Metal Shading Language and cached on device.
- **Hand-aimed weapons.** Guns follow the controller in your hand, with the
  reticle drawn at the gun hand, and health and blood shown as arcs
  around it.
- **PlayStation VR2 Sense controllers, tracked.** Both halves are tracked
  in the room. In menus, point a controller to move the cursor.
- **Gauges on the controllers.** The HUD's side gauges ride on the
  controllers instead of floating in front of your face.
- **A comfortable held camera.** In dialogue and in third person the view
  holds steady around you, so close-ups and chase views never swing with
  your head.
- **Effects that work in stereo.** Disciplines such as Presence and
  Celerity render correctly in each eye.

## No game data included

This repository and every build made from it contain no game data: no
maps, models, textures, sounds, scripts or executables. To play, you need
your own legally obtained, installed copy of the game. The build refuses
to package an app that carries game files.

## Build and install

See [AVP-INSTALL.md](AVP-INSTALL.md) for requirements, building from a
checkout, installing on Apple Vision Pro, and where your game files go.

## Credits

Embrace builds on these open-source projects:

- [Bloodlines Porter](https://github.com/OlehPetrykUA/bloodlines-porter)
  (MIT), the native runtime and build toolchain the port is built on,
  with its recompiler from xboxrecomp by sp00nz (MIT)
- [DXVK](https://github.com/doitsujin/dxvk) (Zlib), its D3D9 shader front end, built
  from the [jm2 fork](https://github.com/jm2/dxvk),
  with [dxbc-spirv](https://github.com/doitsujin/dxbc-spirv) (MIT) and
  [libdisplay-info](https://gitlab.freedesktop.org/emersion/libdisplay-info) (MIT)
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) (Apache-2.0)
- [SDL](https://github.com/libsdl-org/SDL) (Zlib)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg) (LGPL-2.1-or-later), for the
  game's movies

## Licence

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 A. Cervantez.
