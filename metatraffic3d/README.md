# metatraffic3d

The 3D simulator and environments of [MetaTraffic](../README.md). The top-level README covers
the design, the environment APIs, the reward and the measurements; this file maps the code.

| Module | Role |
|---|---|
| `config.py` | `SimConfig`: map, network, world, traffic, taxi and profile settings |
| `profile.py` | samples each city's character (terrain, water, district mix, densities, highway) |
| `roads.py` | lane-level road network: directed segments, lanes, junction turn paths and conflicts, parking |
| `zoning.py`, `heights.py` | districts from smooth fields; building types and heights |
| `terrain.py` | seeded height field, water and embankments |
| `meshes.py`, `world.py` | geometry helpers and city assembly (`build_world`) |
| `traffic/` | microsimulation: `sim.py` (IDM, intersection manager, lane changes, parking), `router.py`, `control.py` (lights, stops), `vehicles.py`, `pedestrians.py`, `events.py` (incidents), `demand.py` |
| `driving/env.py` | `DrivingEnv` (Gymnasium API): ego car, cameras, reward, termination |
| `driving/ego.py` | projects the ego car onto the lane network and logs infractions |
| `driving/cameras.py` | the 8-camera rig |
| `driving/expert.py` | privileged autopilot |
| `driving/collect.py`, `driving/evaluate.py` | demonstration recorder and closed-loop evaluation harness |
| `driving/demo.py`, `driving/gallery.py` | multi-camera videos and weather / city galleries |
| `marl/taxi_env.py`, `marl/baselines.py` | `TaxiEnv` (PettingZoo-style parallel API) and heuristic baselines |
| `backends/mujoco/` | MJCF builder, ego vehicle model, runtime and rendering |
| `backends/isaac/` | USD export and the Isaac Sim runner ([README](backends/isaac/README.md)) |
| `assets/` | Poly Pizza manifest, downloader and loader with procedural fallbacks |
