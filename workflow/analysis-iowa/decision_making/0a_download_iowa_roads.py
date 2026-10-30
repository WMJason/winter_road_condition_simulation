#!/usr/bin/env python3
"""Download Iowa-region major roads only, with bounded, resumable requests."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import osmnx as ox
from osmnx import _overpass
import requests
from shapely.geometry import box, Polygon

ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "iowa_major_roads.graphml"
BBOX_WGS84 = (-96.639, 40.375, -90.140, 43.501)  # west, south, east, north
HIGHWAY_FILTER = '["highway"~"^(motorway|trunk|primary|secondary|motorway_link|trunk_link|primary_link|secondary_link)$"]["access"!~"private|no"]'
ENDPOINTS = ["https://lambert.openstreetmap.de/api",
             "https://overpass.private.coffee/api",
             "https://overpass-api.de/api"]
CACHE_DIR = ROOT / "cache" / "road_download_chunks"
SERVER_TIMEOUT_SECONDS = 120
CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 150
ATTEMPTS_PER_ENDPOINT = 1
MAX_SPLIT_DEPTH = 2
REQUEST_GAP_SECONDS = 3
MAX_RETRY_WAIT_SECONDS = 120
USER_AGENT = "IowaWinterDecisionSupport/1.0 (academic road-network download; OSMnx)"
RETRY_PASS_WAIT_SECONDS = 300
MAX_RETRY_PASS_WAIT_SECONDS = 1800
MAX_DOWNLOAD_PASSES = None  # None: keep retrying until complete or Ctrl+C.


class ChunkDownloadError(RuntimeError):
    def __init__(self, message, can_split):
        super().__init__(message)
        self.can_split = can_split


class PendingRegionsError(RuntimeError):
    pass


class ServerCooldownError(RuntimeError):
    def __init__(self, seconds):
        super().__init__(f"Server requests a {seconds}s cooldown; cached downloads are retained.")
        self.seconds=seconds


def cache_path(query):
    return CACHE_DIR / (hashlib.sha256(query.encode()).hexdigest() + ".json.gz")


def save_response(query, value):
    path=cache_path(query)
    temporary=path.with_suffix(path.suffix+".tmp")
    with gzip.open(temporary,"wt",compresslevel=1) as handle:
        json.dump(value,handle,separators=(",", ":"))
    temporary.replace(path)


def checked_response(value):
    if not isinstance(value, dict) or not isinstance(value.get("elements"), list):
        raise ValueError("Server did not return an OSM elements list")
    if value.get("remark"):
        raise ValueError(f"Server reported an incomplete query: {value['remark']}")
    return value


def fetch_chunk(query, index, total, session):
    """Cache by query, not server, so retries retain successful chunks."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(query)
    if path.exists():
        try:
            with gzip.open(path, "rt") as handle:
                value = checked_response(json.load(handle))
            print(f"[{index}/{total}] Cache hit: {len(value['elements']):,} OSM elements", flush=True)
            return value
        except (OSError, EOFError, ValueError):
            print(f"[{index}/{total}] Invalid cached response; downloading again", flush=True)
    failures = [];can_split=False
    for endpoint in ENDPOINTS:
        for attempt in range(1, ATTEMPTS_PER_ENDPOINT + 1):
            print(f"[{index}/{total}] Downloading from {endpoint}, attempt {attempt}/{ATTEMPTS_PER_ENDPOINT}", flush=True)
            pause = 10
            try:
                # Direct requests avoid OSMnx status polling and DNS/IP pinning.
                url = endpoint.rstrip("/") + "/interpreter"
                response = session.get(url, params={"data": query},
                                       timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
                if response.status_code == 414:
                    response = session.post(url, data={"data": query},
                                            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
                if response.status_code in {429, 503, 504}:
                    pause = 60
                    retry = response.headers.get("Retry-After", "")
                    if retry.isdigit():
                        pause = max(pause, int(retry))
                    if pause > MAX_RETRY_WAIT_SECONDS:
                        raise ServerCooldownError(pause)
                response.raise_for_status()
                value = checked_response(response.json())
                save_response(query,value)
                print(f"[{index}/{total}] Saved {len(value['elements']):,} OSM elements", flush=True)
                time.sleep(REQUEST_GAP_SECONDS)
                return value
            except (requests.RequestException, ValueError) as error:
                response=getattr(error,"response",None)
                can_split |= isinstance(error,requests.ReadTimeout) or (
                    response is not None and response.status_code==504) or "incomplete query" in str(error)
                detail = f"{endpoint}, attempt {attempt}: {str(error)[:240]}"
                failures.append(detail)
                print(f"  Request failed: {detail}\n  Waiting {pause}s before the next attempt", flush=True)
                time.sleep(pause)
    raise ChunkDownloadError(f"Chunk {index}/{total} failed after bounded retries. "
                       "Completed chunks are saved; rerun to resume.\n" + "\n".join(failures),can_split)


def fetch_region(coordinates, settings, road_filter, index, total, session, depth=0):
    query=f"{settings};(way{road_filter}(poly:{coordinates!r});>;);out;"
    path=cache_path(query)
    marker=path.with_suffix(".split.json")
    if path.exists() or not marker.exists():
        try:
            return fetch_chunk(query,index,total,session)
        except ChunkDownloadError as error:
            if not error.can_split or depth>=MAX_SPLIT_DEPTH:raise
    if depth>=MAX_SPLIT_DEPTH:
        raise RuntimeError(f"Region {index} still fails at maximum subdivision; rerun later. Cached regions are retained.")
    # OSMnx supplies alternating latitude/longitude values. Intersect quadrants
    # with this polygon, then union OSM IDs to retain boundary-crossing ways.
    numbers=list(map(float,coordinates.split()))
    polygon=Polygon([(numbers[i+1],numbers[i]) for i in range(0,len(numbers),2)])
    west,south,east,north=polygon.bounds
    mx,my=(west+east)/2,(south+north)/2
    regions=[]
    for bounds in [(west,south,mx,my),(mx,south,east,my),(west,my,mx,north),(mx,my,east,north)]:
        intersection=polygon.intersection(box(*bounds))
        parts=[intersection] if intersection.geom_type=="Polygon" else getattr(intersection,"geoms",[])
        regions.extend(part for part in parts if part.geom_type=="Polygon" and part.area>1e-12)
    if len(regions)<2:raise RuntimeError("Unable to subdivide failed query safely")
    marker.write_text(json.dumps({"depth":depth,"child_regions":len(regions)}))
    print(f"[{index}/{total}] Splitting slow region into {len(regions)} smaller queries (depth {depth+1}/{MAX_SPLIT_DEPTH})",flush=True)
    elements={};result=None
    for child,region in enumerate(regions,1):
        text=" ".join(f"{lat:.7f} {lon:.7f}" for lon,lat in region.exterior.coords)
        response=fetch_region(text,settings,road_filter,f"{index}.{child}",total,session,depth+1)
        if result is None:result={key:value for key,value in response.items() if key!="elements"}
        for element in response["elements"]:elements[element["type"],element["id"]]=element
    result["elements"]=list(elements.values())
    save_response(query,result)
    print(f"[{index}/{total}] Completed subdivided region: {len(elements):,} unique OSM elements",flush=True)
    return result


def bounded_network_download(polygon, network_type, custom_filter):
    """Replace transport only; retain OSMnx graph construction and topology."""
    filters = custom_filter if isinstance(custom_filter, list) else [
        custom_filter or _overpass._get_network_filter(network_type)]
    settings = _overpass._make_overpass_settings()
    polygons = _overpass._make_overpass_polygon_coord_strs(polygon)
    total = len(polygons) * len(filters)
    print(f"Downloading road network in {total} resumable requests (no server-status polling)", flush=True)
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        index = 0;pending=[]
        report=CACHE_DIR.parent/"pending_road_regions.json"
        report.parent.mkdir(parents=True,exist_ok=True)
        for coordinates in polygons:
            for road_filter in filters:
                index += 1
                try:
                    value=fetch_region(coordinates,settings,road_filter,index,total,session)
                except ChunkDownloadError as error:
                    pending.append({"region":index,"error":str(error)})
                    report.write_text(json.dumps({"total_regions":total,"checked_regions":index,
                        "pending_regions":pending,"graph_exported":False},indent=2))
                    print(f"[{index}/{total}] Deferred unavailable region; proceeding to the next region. "
                          "Cached subregions are retained.",flush=True)
                    continue
                yield value
        report.write_text(json.dumps({"total_regions":total,"checked_regions":index,
            "pending_regions":pending,"graph_exported":False},indent=2))
        if pending:
            missing=", ".join(str(row["region"]) for row in pending)
            raise PendingRegionsError(f"Download pass finished. Pending regions: {missing}. "
                               f"Details: {report}. The next pass will reuse all completed regions. "
                               "No incomplete GraphML was exported; any older GraphML is not this completed download.")


def main():
    if not ox.__version__.startswith("2."):
        raise RuntimeError("Use your osmnx_env environment with OSMnx 2.x")
    ox.settings.log_console = True
    ox.settings.requests_timeout = SERVER_TIMEOUT_SECONDS
    print("Downloading Iowa major roads only; output is GraphML.", flush=True)
    # Installed OSMnx 2.x hook: simplification, one-way handling, parallel edges,
    # buffering and polygon clipping remain OSMnx's, not hand-built topology.
    original = _overpass._download_overpass_network
    _overpass._download_overpass_network = bounded_network_download
    try:
        number=0;wait=RETRY_PASS_WAIT_SECONDS
        while True:
            number+=1
            print(f"\nDownload pass {number}. Press Ctrl+C to stop; saved regions are retained.",flush=True)
            try:
                graph = ox.graph_from_polygon(box(*BBOX_WGS84), network_type="drive",
                                              custom_filter=HIGHWAY_FILTER, simplify=True, retain_all=True)
                break
            except (PendingRegionsError,ServerCooldownError) as error:
                if MAX_DOWNLOAD_PASSES is not None and number>=MAX_DOWNLOAD_PASSES:
                    raise RuntimeError(f"Reached {MAX_DOWNLOAD_PASSES} download passes; rerun to resume.") from error
                delay=max(wait,error.seconds) if isinstance(error,ServerCooldownError) else wait
                print(f"{error}\nWaiting {delay/60:.1f} minutes before retrying automatically.",flush=True)
                time.sleep(delay)
                wait=min(wait*2,MAX_RETRY_PASS_WAIT_SECONDS)
    finally:
        _overpass._download_overpass_network = original
    if not graph.is_directed() or not graph.number_of_edges():
        raise ValueError("Downloaded graph is not a nonempty directed road network")
    west, south, east, north = BBOX_WGS84
    if any(not (west-.02 <= float(d["x"]) <= east+.02 and south-.02 <= float(d["y"]) <= north+.02)
           for _, d in graph.nodes(data=True)):
        raise ValueError("Graph coordinates are outside the requested Iowa region")
    temporary = OUTPUT_FILE.with_suffix(".graphml.tmp")
    ox.save_graphml(graph, temporary)
    temporary.replace(OUTPUT_FILE)
    report=CACHE_DIR.parent/"pending_road_regions.json"
    if report.exists():
        state=json.loads(report.read_text());state.update(graph_exported=True,output_file=str(OUTPUT_FILE))
        report.write_text(json.dumps(state,indent=2))
    print(f"Done: {len(graph):,} nodes; {graph.number_of_edges():,} directed edges\nSaved: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. Completed downloads are saved; run this script again to resume.",flush=True)
