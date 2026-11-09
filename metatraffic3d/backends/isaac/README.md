# Isaac Sim backend

The same city as the MuJoCo backend, as a USD stage for NVIDIA Isaac Sim, plus a script that runs traffic in it.

| File | What it does | Status |
|---|---|---|
| `usd_export.py` | `export_usd(world, "city.usd")` writes the stage (needs only `usd-core`) | verified with usd-core |
| `mirror.py` | `StageMirror` / `KinematicEgo` move cars, parked cars, people, incident props, lamps and the ego on the stage (plain `pxr`) | verified with usd-core |
| `run_isaac.py` | Isaac Sim script: app, stage loading, frame loop, camera capture with Replicator | **untested** |

## Use

```bash
# anywhere (macOS is fine): export the city
pip install usd-core
python -m metatraffic3d.backends.isaac.usd_export --seed 4 --out city.usd      # .usda for text

# on a Linux/Windows machine with an NVIDIA GPU and Isaac Sim 4.x/5.x
./python.sh -m pip install shapely pillow       # plus this repo on the path (pip install -e .)
./python.sh metatraffic3d/backends/isaac/run_isaac.py --usd city.usd --seed 4 --seconds 60 \
    --headless --out frames/ --cameras front_main,pillar_left,pillar_right,rear,chase
```

`--seed` seeds the traffic. The map comes from the `SimConfig` that the exporter stores in the stage, so the runner rebuilds the same road network without the meshes.

## Stage layout

`/World` is the default prim. Units are metres, Z is up.

- `/World/City/<group>/<mesh>`: static meshes, one shared `UsdPreviewSurface` per colour, flat normals on buildings and props
- `/World/Props/<asset>`: one `PointInstancer` per prop asset (trees, street lights, stop signs)
- `/World/Signals/head_<i>/{red,yellow,green}`: emissive lamp spheres the runner shows and hides; `metatraffic:node`/`deIn` name the approach
- `/World/Prototypes/<type>`: invisible, instanceable vehicles, `person` and incident props; paint reads the `paintColor` primvar
- `/World/Prototypes/ego`: car plus the 8 `TESLA_RIG` cameras (focal length from hfov, 16:9 aperture)
- `/World/Physics`: `UsdPhysics.Scene` (gravity −Z 9.81) and an invisible ground collider (terrain grid, or a coarse plane when flat) with a friction material
- Semantic class labels for Replicator, sun and dome light

## What was verified here, and how

On macOS with `usd-core` 0.26 (`tests/test_isaac_usd.py`, 11 tests):

- The exported stage reopens. Mesh count, faces and vertices match `World.scene`. Materials and colours match. The bounding box matches `world.bounds`.
- Prop positions follow the terrain, including on a swapped-in hilly terrain. Instance orientation and scale match.
- Lamps and signal attributes, prototype face counts, and the physics scene, ground and friction are checked.
- Camera FOV is checked with `Gf.Camera`, and camera position and view direction are checked against the rig.
- Binary crate output, baked-prop mode (`instancing=False`) and the config round trip are checked.
- `StageMirror` places live `TrafficSim` cars exactly at `sim.pose` on native instances, with their own paint colours. Each signal head lights at most one lamp, and the ego cameras are real prims. Writes go to the session layer only, so the file stays unchanged.
- The default-world export passes all 30 `UsdValidation` validators that ship with usd-core, with 0 issues.

## Not verified (needs Isaac Sim)

- Everything in `run_isaac.py`: `SimulationApp` config keys, `omni.usd` stage loading, the `rep.orchestrator.step` arguments, annotator output formats and the viewport camera call. Expect small API fixes on the first run.
- **RTX rendering.** Untested: whether RTX applies inherited per-instance primvars (`paintColor`, light bars) inside native instances. If cars render in their default colour, use `StageMirror(..., instanceable=False)`. The light and emissive intensities (sun 3000, dome 1000, lamps ×4) are guesses.
- **Semantic labels.** They are written as raw `SemanticsAPI` and `SemanticsLabelsAPI` attributes. Untested in Replicator.
- **PhysX.** Untested whether PhysX parses the invisible ground mesh collider. `PhysxSchema` extras are only added when exporting from Isaac's Python.

## Known limitations

- The ego is kinematic: it rides a background car from `TrafficSim`. A PhysX vehicle (`PhysxVehicleAPI`) with `EgoTracker` and an expert driver is future work. Only the ground has colliders.
- No fog or rain rendering, headlights or street-light lights. Weather only changes the sun, sky and ground friction.
- Pinhole cameras only, as in MuJoCo.
- Parked cars are pooled near the ego (within 200 m) and pedestrians are the nearest 120, as in the MuJoCo runtime.
