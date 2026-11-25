# Leather glove hands

Original first person hands: a worn dark leather glove with covered
fingertips, a flared cuff, oxblood stitching along the finger and side
seams, three stitched points on the back and scuffing on the knuckles.
Built entirely by `build_gloves.py`; nothing here comes from a game.

## Files

| File | What it is |
| --- | --- |
| `glove_right.usdz`, `glove_left.usdz` | mesh, skeleton, skin weights, material and textures |
| `poses.json` | the nine poses as per joint quaternions, plus the rest skeleton |
| `textures/` | 1024 base colour (sRGB), roughness and normal maps, shared by both hands |
| `build_gloves.py` | the generator; it also writes `gloves.blend` (both hands, both rigs, the poses as pose library actions) and Cycles stills under `preview/` |

## Convention (the USDZ files and poses.json)

* Metres, Y up, right handed.
* Origin at the wrist joint.
* Fingers point along negative Z, the palm faces negative Y, the back of
  the hand faces positive Y.
* The right thumb is on negative X, the left thumb on positive X. The left
  hand is an exact mirror of the right across X.
* The generated `gloves.blend` works in Blender's Z up frame (fingers along +Y, palm
  facing -Z). The exporter rotates the data itself, so the USDZ points
  are already in the frame above and carry no extra root rotation.

## Mesh

6272 triangles and 3138 vertices per hand in the USDZ (triangulated for
the renderer). The .blend keeps the 3136 quads, with edge loops around
every knuckle. The UVs follow a real glove pattern: a back piece, a palm
piece and the cuff lining, split along the finger and hand sides. The left
hand reuses the right hand's UVs mirrored, so its tangent frames have the
opposite handedness; compute tangents with a bitangent sign (MikkTSpace
or equivalent) and the one normal map serves both hands. The normal map
is +Y (OpenGL style).

## Skeleton

26 joints named after visionOS `HandSkeleton.JointName`: `wrist` (root),
`forearmWrist`, `thumbKnuckle`, `thumbIntermediateBase`,
`thumbIntermediateTip`, `thumbTip`, and for index, middle, ring and
little `<finger>FingerMetacarpal`, `Knuckle`, `IntermediateBase`,
`IntermediateTip`, `Tip`. `forearmWrist` and the four metacarpals hang
from `wrist`. The five `Tip` joints sit at the fingertips and carry no
weight; the other 21 deform. At most four influences per vertex.

Each joint's frame: Y runs along the bone toward the next joint, Z points
out of the back of the hand (the nail side), X completes a right handed
frame. Negative rotation about X curls a finger toward the palm. This
frame is the same in Blender and in the USDZ.

The rest pose is a relaxed open hand with a slight natural curl.

## poses.json

`hands.right` and `hands.left` each hold:

* `joints`: names in the same order as the USD skeleton.
* `parents`: index of each joint's parent, -1 for the root.
* `rest`: each joint's rest transform relative to its parent, `t` in
  metres and `r` as a quaternion. These equal the USD `restTransforms`.
* `poses`: for `rest`, `open`, `relaxed`, `trigger`, `grip`, `fist`,
  `point`, `pistol_grip`, `rifle_grip` and `melee_grip`, a quaternion per
  joint in two forms:
  * `local`: the joint rotation relative to its parent, ready to use as
    the rotation of the joint's local transform (keep the rest `t`).
  * `delta`: the rotation relative to rest in the joint's own frame, so
    `local = rest.r * delta`.

Quaternions are `[x, y, z, w]` with `w` kept non negative. To blend two
poses, slerp per joint (or nlerp, then normalise). `grip` curls the
middle, ring and little fingers and leaves the index free. `rifle_grip`
is the support hand around a fore end. `melee_grip` is a tight hammer
grip with the thumb over the index.

## Checks

* `usdchecker --arkit` passes on both files.
* Every finger was bent to 90, 105 and 70 degrees at its three joints and
  the thumb to 45, 60 and 75: no vertex loses its weights and nothing
  opens a hole (the skin is one closed shell, so it cannot tear). The
  only folding is a few faces inside the palm side crease of a fully
  bent joint, where leather bunches anyway; the back of the hand stays
  clean.
* The posed left hand matches the mirrored posed right hand to within a
  thousandth of a millimetre in all nine poses.

## Rebuilding

Blender 4.2, run headless, one stage at a time:

    B=/Applications/Blender.app/Contents/MacOS/Blender
    for stage in model check bake export preview; do
      $B --background --factory-startup --python build_gloves.py -- $stage .
    done

Intermediate files go to `/tmp/glove_build` (set `GLOVE_WORK` to move
them). `GLOVE_SAMPLES` sets the preview sample count.
