#!/usr/bin/env python3
"""Prepare <=500 m physical road cells and one midpoint support per cell."""

import json
import hashlib
import tempfile
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import substring, transform

# --- Configuration & Paths ---
ROOT = Path(__file__).resolve().parent
INPUT_GRAPH = ROOT / "iowa_major_roads.graphml"
OUTPUT_DIR = ROOT / "shared_cache"

SUPPORT_SPACING_M = 500.0  # 500m segment sampling interval
METRIC_CRS = "EPSG:32615"  # UTM Zone 15N for Iowa (metric projection)

def load_and_process_roads():
    if not np.isfinite(SUPPORT_SPACING_M) or SUPPORT_SPACING_M <= 0:
        raise ValueError("SUPPORT_SPACING_M must be positive")
    if not INPUT_GRAPH.exists():
        raise FileNotFoundError(
            f"Graph file '{INPUT_GRAPH}' not found. "
            "Please run '0a_download_iowa_roads.py' first!"
        )

    print(f"Loading road network from {INPUT_GRAPH}...", flush=True)
    raw = nx.read_graphml(INPUT_GRAPH, node_type=str, force_multigraph=True)
    if not raw.is_directed() or not raw.number_of_edges():
        raise ValueError("A nonempty directed road graph is required")
    coordinates=np.array([(float(d['x']),float(d['y'])) for _,d in raw.nodes(data=True)])
    if not ((coordinates[:,0]>=-98)&(coordinates[:,0]<=-89)&(coordinates[:,1]>=39)&(coordinates[:,1]<=45)).all():
        raise ValueError("Road graph is not in the Iowa region. Re-run corrected 0a; no support files were overwritten.")

    # Coordinate transformers
    to_metric = Transformer.from_crs(4326, METRIC_CRS, always_xy=True)
    to_geo = Transformer.from_crs(METRIC_CRS, 4326, always_xy=True)

    physical_roads = set()
    support_points = []

    # 1. Process Physical Links and Sample Support Points
    print("Sampling 500m midpoint support locations along Iowa road links...", flush=True)
    
    for u, v, key, data in raw.edges(keys=True, data=True):
        if "geometry" in data:
            from shapely.wkt import loads
            line = loads(data["geometry"]) if isinstance(data["geometry"], str) else data["geometry"]
        else:
            line = LineString([
                (float(raw.nodes[u]["x"]), float(raw.nodes[u]["y"])),
                (float(raw.nodes[v]["x"]), float(raw.nodes[v]["y"]))
            ])

        if line.is_empty or line.geom_type!="LineString":
            continue

        # Project geometry to UTM metric scale
        metric_line = transform(to_metric.transform, line)
        length_km = float(metric_line.length / 1000.0)
        if length_km <= 0:
            continue

        coords=tuple(metric_line.coords)
        canonical=min(coords,tuple(reversed(coords)))
        physical_id="P_"+hashlib.sha256(json.dumps([sorted([u,v]),canonical],sort_keys=True,default=str).encode()).hexdigest()[:20]

        if physical_id not in physical_roads:
            physical_roads.add(physical_id)

            # --- Sample 500m Support Segments ---
            geom_length = metric_line.length
            for order, start in enumerate(np.arange(0, geom_length, SUPPORT_SPACING_M)):
                end = min(start + SUPPORT_SPACING_M, geom_length)
                midpoint = metric_line.interpolate((start + end) / 2.0)
                lon, lat = to_geo.transform(midpoint.x, midpoint.y)

                support_points.append({
                    "segment_id": f"{physical_id}_S{order:05d}",
                    "physical_id": physical_id,
                    "segment_order": order,
                    "length_km": (end - start) / 1000.0,
                    "x_m": midpoint.x,
                    "y_m": midpoint.y,
                    "longitude": lon,
                    "latitude": lat,
                    "geometry_wkt_metric": substring(metric_line,float(start),float(end)).wkt,
                })

    df_points = pd.DataFrame(support_points)
    if df_points.empty:
        raise ValueError("No usable road geometries were found")

    print(
        f"Generated {len(df_points):,} 500m support segments across "
        f"{len(physical_roads):,} physical roads.",
        flush=True,
    )

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    output=OUTPUT_DIR/"0b_road_support_points.csv.gz"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR,suffix=".tmp",delete=False) as handle:
        temporary=Path(handle.name)
    try:
        df_points.to_csv(temporary,index=False,compression={"method":"gzip","compresslevel":1,"mtime":0})
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = {
        "graph_file": str(INPUT_GRAPH),
        "graph_signature": [INPUT_GRAPH.stat().st_size,INPUT_GRAPH.stat().st_mtime_ns],
        "metric_crs": METRIC_CRS,
        "support_spacing_m": SUPPORT_SPACING_M,
        "support_segment_count": len(df_points),
        "physical_road_count": len(physical_roads),
        "total_network_length_km": float(df_points["length_km"].sum()),
        "support_points_file": str(output),
        "point_definition": f"midpoint of each <={SUPPORT_SPACING_M:g}m road cell; shorter final cell at each link end",
    }
    (OUTPUT_DIR / "0b_support_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nMetadata:\n", json.dumps(metadata, indent=2), flush=True)

    print(f"Saved support points: {output}",flush=True)


if __name__ == "__main__":
    load_and_process_roads()
