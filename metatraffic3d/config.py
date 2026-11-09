"""All tunable settings in one place. Units: metres, seconds, m/s."""

from __future__ import annotations

from dataclasses import dataclass, field

from .profile import ProfileConfig


@dataclass(frozen=True)
class RoadClass:
    """Cross-section of one road class, per direction of travel (right-hand traffic).

    A "sub-road" has 4 lanes in total: per direction one driving lane and one parking lane at
    the kerb. A main road has 8 driving lanes (4 per direction), a median and no parking.
    """

    name: str
    drive_lanes: int  # driving lanes per direction
    lane_width: float
    parking: bool  # one kerb-side parking lane per direction
    park_width: float
    sidewalk: float
    speed_limit: float  # m/s
    priority: int  # higher = right of way at intersections
    median: float = 0.0  # width of the strip between the two directions

    @property
    def half_width(self) -> float:
        return self.median / 2 + self.drive_lanes * self.lane_width + (self.park_width if self.parking else 0.0)

    def lane_offset(self, k: int) -> float:
        """Distance of driving lane ``k`` (0 = kerb side) to the right of the centre line."""
        return self.median / 2 + (self.drive_lanes - k - 0.5) * self.lane_width

    @property
    def park_offset(self) -> float:
        return self.median / 2 + self.drive_lanes * self.lane_width + self.park_width / 2

    @property
    def is_sub_road(self) -> bool:
        return self.parking


MAIN = RoadClass("main", 4, 3.5, False, 0.0, 3.0, 50 / 3.6, 2, median=1.0)
MAJOR = RoadClass("major", 1, 3.25, True, 2.5, 2.5, 40 / 3.6, 1)
MINOR = RoadClass("minor", 1, 3.25, True, 2.5, 2.5, 30 / 3.6, 0)
HIGHWAY = RoadClass("highway", 2, 3.5, False, 0.0, 0.0, 80 / 3.6, 3, median=1.0)  # experimental viaduct
ROAD_CLASSES = {c.name: c for c in (MAIN, MAJOR, MINOR, HIGHWAY)}
DECK_HEIGHT = 9.0  # highway deck above street level
RAMP_LENGTH = 110.0  # ramp at each end of the highway


@dataclass
class MapConfig:
    """Which 2D city to build (passed to mapgen) and how big a map unit is."""

    seed: int = 4
    width: float = 500.0  # mapgen view size in SVG units (see mapgen.CityConfig)
    height: float = 320.0
    zoom: float = 0.638
    metres_per_unit: float = 3.7  # 1 mapgen world unit; gives ~74 m between minor roads
    water: bool = True


@dataclass
class NetworkConfig:
    snap_distance: float = 9.0  # merge intersections closer than this
    min_dead_end: float = 30.0  # drop dangling stubs shorter than this
    corner_margin: float = 1.5  # extra clearance when trimming lanes at intersections
    max_trim: float = 28.0
    dead_end_trim: float = 9.0  # room for U-turns at dead ends


@dataclass
class ParkingConfig:
    spot_length: float = 6.0
    end_clearance: float = 5.0  # no parking this close to an intersection
    occupancy: tuple[float, float] = (0.35, 0.75)  # initial fill per kerb, drawn uniformly
    max_occupancy: float = 0.8  # a kerb never fills beyond this ("the road is never full")
    churn_per_hour: float = 0.6  # fraction of parked cars that leave per hour (and are replaced)


@dataclass
class BuildingConfig:
    ground_floor: float = 4.5
    floor_height: float = 3.2
    max_floors: int = 60
    downtown_sigma: float = 0.18  # downtown spread as a fraction of the map diagonal
    seed_offset: int = 101


@dataclass
class WorldConfig:
    river_depth: float = 3.5  # water surface below street level
    sea_depth: float = 1.2
    coast_strip: float = 15.0  # map units of land kept between sea and the coast road
    trees_per_100m2: float = 0.9
    tree_clearance: float = 2.5
    street_trees: bool = True
    street_tree_spacing: float = 18.0


@dataclass
class TrafficConfig:
    dt: float = 1.0
    num_npc: int = 120
    # Intelligent Driver Model
    max_accel: float = 1.5
    comfort_decel: float = 2.5
    time_headway: float = 1.2
    min_gap: float = 2.0
    accel_exponent: float = 4.0
    lateral_accel: float = 2.5  # sets turn speeds
    request_distance: float = 35.0  # ask the intersection for a slot this far out
    starvation_time: float = 25.0  # after this wait, a request blocks newer conflicting ones
    stuck_timeout: float = 150.0  # an NPC stuck this long is removed (like SUMO's teleport)
    congestion_ema: float = 0.15
    downtown_bias: float = 1.0  # trip ends weighted by floor area ** bias
    num_pedestrians: int = 150
    jaywalk_rate: float = 0.04  # chance per block walked that a pedestrian crosses mid-block
    reckless_share: float = 0.15  # jaywalkers who do not wait for a gap
    grade_effect: float = 0.3  # share of g*sin(slope) that NPC drivers do not compensate


@dataclass
class TaxiConfig:
    num_taxis: int = 5
    episode_seconds: float = 1800.0
    request_rate_per_min: float = 2.0
    patience: tuple[float, float] = (240.0, 600.0)  # seconds a passenger waits before leaving
    min_trip: float = 500.0
    max_request_options: int = 6
    max_route_options: int = 3
    route_slack: float = 1.5  # keep route options whose ETA <= best * slack + 60 s
    boarding_time: float = 15.0
    alighting_time: float = 10.0
    wait_action_seconds: float = 30.0
    fare_base: float = 1.0
    fare_per_km: float = 0.5
    wait_penalty_per_min: float = 0.05
    drive_cost_per_min: float = 0.1
    abandon_penalty: float = 1.0
    double_park_penalty: float = 0.2
    team_reward: float = 0.0  # 0 = own reward only, 1 = fleet average
    pickup_search: float = 45.0  # look for a free kerb spot within this distance


@dataclass
class SimConfig:
    map: MapConfig = field(default_factory=MapConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    parking: ParkingConfig = field(default_factory=ParkingConfig)
    buildings: BuildingConfig = field(default_factory=BuildingConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    taxi: TaxiConfig = field(default_factory=TaxiConfig)
