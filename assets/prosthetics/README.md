# Prosthetic Assets

This folder vendors simulator assets needed by the live inference runtime.

Current contents:
- `dexhandv2/urdf/dexhandv2_right.urdf`
- `dexhandv2/urdf/dexhandv2_cobot_right.urdf`
- `dexhandv2/meshes/right/*.stl`
- `dexhandv2/meshes/cobot_right/*.stl`

These files were copied from the older local `SpikeFormerMyo` project so this
repo can resolve DexHand simulator assets without depending on a sibling checkout.

The live MuJoCo runtime rewrites the original `package://...` mesh paths into
local absolute mesh paths before loading the hand model.
