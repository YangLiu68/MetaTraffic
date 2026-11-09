<p align="center">
  <img src="./logo.png" width="80" style="border-radius: 12px;" />
</p>

<h1 align="center">MetaTraffic</h1>
<p align="center">
  A multi-agent environment runtime for studying coordination under partial observability
</p>

MetaTraffic is a multi-agent reinforcement learning environment runtime for training and evaluating decentralized policies under realistic perception constraints.

The system focuses on **environment design, simulation dynamics, and agent interaction**, rather than model architecture alone. It is designed as a lightweight prototype of an agent training environment for real-world tasks.
<p align="center">
  <img src="./demo.gif" width="600"/>
</p>


## MetaTraffic 3D

[`metatraffic3d/`](metatraffic3d/README.md) turns the generated 2D maps into seeded 3D cities
(terrain, districts, buildings, water, bridges, traffic lights, pedestrians, incidents, weather)
with two simulator backends (MuJoCo, tested; Isaac Sim via USD, export tested). It adds a
camera-only end-to-end driving setup (physical car with direct steer/throttle/brake/gear
control, 8-camera rig, time-aware transformer policy pre-trained by imitation and fine-tuned
with RL) and a multi-agent taxi fleet trained with MAPPO.

```bash
pip install -e ".[mujoco,learn,assets]"
python -m metatraffic3d drive-demo --seed 7 --weather rain --seconds 60 --out demo.mp4
```

<p align="center"><img src="docs/images/cities.jpg" width="720"/></p>

## Overview

Most RL environments assume either:

* single-agent control, or
* fully observable global state

MetaTraffic explores a more realistic setting:

> Multiple agents interacting in a shared world, with limited local perception and competing objectives.

This setup is closer to real-world systems such as autonomous driving or robotic coordination.

---

## Core Idea

Instead of exposing full world state, each agent observes a **local spatial representation** of the environment.

This is implemented as a rasterized road mask centered on the agent, functioning as a lightweight **occupancy grid**, similar to LiDAR or radar perception in real systems.

This forces agents to learn:

* local decision making
* implicit coordination
* behavior under partial observability

---

## System Architecture

```
Map (SVG)
  → Road Graph Generation
    → Rasterized Mask (World Representation)
      → Physics Engine (Vehicle Dynamics)
        → Environment Runtime (DrivingEnv)
          → RL Training Loop
            ↘ Visualization / Debugging
```

---

## Environment Runtime

The system implements a Gym-like interface:

* `reset()` → initialize episode with randomized start/goal
* `step(action)` → advance simulation

Returns:

* observation
* reward
* done

This design separates **world simulation** from **agent interaction**, making the environment reusable for different training setups.

---

## Observation Model (Perception)

Each agent receives:

* 32×32 local spatial mask (road / obstacle encoding)
* normalized velocity
* heading direction (cos, sin)
* relative displacement to goal

This enforces **partial observability** and removes access to global state.

---

## Simulation Dynamics

Each episode consists of:

* generated or selected map
* randomized agent spawn points
* individual destinations

At every step, the system:

1. updates world state
2. applies agent actions
3. resolves collisions and constraints
4. computes rewards
5. returns next observations

---

## Reward Design

The environment includes stabilization mechanisms:

* potential-based shaping
* progress rewards
* collision penalties
* off-road penalties with temporal tracking
* stuck detection using history windows

These are critical for maintaining learning stability in multi-agent settings.

---

## Key Challenges Explored

### Coordination under Partial Observability

* limited local perception
* delayed reactions
* emergent congestion

---

### Scaling Multi-Agent Interaction

* collision frequency grows non-linearly
* coordination instability
* noisy reward signals

---

### Environment-Induced Instability

* procedural map variation
* distribution shift
* inconsistent optimal policies

---

## Training

* Algorithm: MADDPG
* Continuous control
* Multi-agent training
* Parallel rollout support

The focus is not the algorithm itself, but how **environment design shapes learning dynamics**.

---

## Debugging & Visualization

* real-time simulation rendering
* collision inspection
* trajectory visualization
* shared pipeline between training runtime and web interface

This enables consistent behavior across simulation and visualization environments.

---

## Map Generation

The road network generation is adapted from:
https://github.com/ProbableTrain/MapGenerator

We extended it to function as part of the environment runtime:

* integrated into the environment reset pipeline
* converted vector maps into rasterized masks for fast spatial queries
* unified usage across Python training and web visualization

This transforms the generator from a standalone tool into a reusable simulation component.

The generator has been ported to Python in [`mapgen/`](mapgen/) (the original TypeScript is
in `MapGenerator/`). It writes the same SVG format, so the environment can load its maps directly:

```bash
pip install -r mapgen/requirements.txt
python -m mapgen --seed 7 -o map.svg --png map.png
```

---

## Limitations

* not distributed across multiple machines
* limited agent communication mechanisms
* not benchmarked against standard MARL suites
* scaling beyond current agent count requires further optimization

---

## Why This Matters

Real-world AI systems operate under:

* partial observability
* noisy perception
* shared environments
* interacting agents

MetaTraffic provides a controlled environment to study these properties, bridging simulation and real-world system constraints.

---

## Repository Structure

```
RL/
  env.py
  physics.py
  train_parallel.py
  train_single.py

Web/
  visualization / interface
```

---

## Acknowledgements

Road generation is adapted from:
https://github.com/ProbableTrain/MapGenerator

Extended for environment integration and cross-runtime usage.
