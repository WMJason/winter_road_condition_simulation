#!/usr/bin/env python3
"""Attach USGS 3DEP terrain elevation to existing road supports and selected RWIS.

No roads are downloaded or resegmented. Source files are preserved. Cached DEM
tiles are the default; batched point queries with SQLite caching are optional.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
SERVICE="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples"


def resolve(path):
    p=Path(path).expanduser()
    return p if p.is_absolute() else ROOT/p


def fingerprint(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for data in iter(lambda:stream.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def payload(points):
    return {"f":"json","geometryType":"esriGeometryMultipoint",
        "geometry":json.dumps({"points":np.asarray(points).tolist(),"spatialReference":{"wkid":4326}},separators=(",",":")),
        "returnFirstValueOnly":"true","interpolation":"RSP_BilinearInterpolation",
        "outFields":"VerticalDatum"}


def decode_samples(response,points):
    if "error" in response:
        raise ValueError(f"Elevation service error: {response['error']}")
    points=np.asarray(points,float)
    result={}
    for sample in response.get("samples",[]):
        index=int(sample["locationId"])
        if index<0 or index>=len(points) or index in result:
            raise ValueError("Invalid or repeated sample locationId")
        location=sample["location"]
        if location.get("spatialReference",{}).get("latestWkid",location.get("spatialReference",{}).get("wkid"))!=4326:
            raise ValueError("Unexpected response CRS; refusing incorrect coordinate matching")
        if not np.allclose([location["x"],location["y"]],points[index],atol=1e-7,rtol=0):
            raise ValueError("Returned elevation location does not match requested point")
        value=float(sample["value"])
        if not np.isfinite(value) or not -500<=value<=9000:
            raise ValueError(f"NoData/implausible elevation: {value}")
        result[index]=(value,float(sample.get("resolution",np.nan)),
                       str(sample.get("attributes",{}).get("VerticalDatum","not reported")))
    if len(result)!=len(points):
        raise ValueError(f"Only {len(result)} of {len(points)} elevation samples returned")
    return [result[i] for i in range(len(points))]


def query(session,points,config):
    parameters=payload(points)
    prepared=requests.Request("GET",config["service_url"],params=parameters).prepare()
    if len(prepared.url)>config["max_url_characters"] and len(points)>1:
        middle=len(points)//2
        return query(session,points[:middle],config)+query(session,points[middle:],config)
    last=None
    for attempt in range(config["attempts"]):
        try:
            response=session.get(config["service_url"],params=parameters,timeout=(15,config["read_timeout_seconds"]))
            response.raise_for_status()
            result=decode_samples(response.json(),points)
            time.sleep(config["request_gap_seconds"])
            return result
        except (requests.RequestException,ValueError,KeyError) as error:
            last=error
            if attempt+1<config["attempts"]:
                time.sleep(min(30,2**(attempt+1)))
    status=getattr(getattr(last,"response",None),"status_code",None)
    # Do not recursively multiply requests during a service outage.
    splittable=(status in [413,414] or (isinstance(last,ValueError) and
        (str(last).startswith("Only ") or str(last).startswith("NoData/") or
         "sample limit" in str(last).lower())))
    if len(points)>1 and splittable:
        middle=len(points)//2
        return query(session,points[:middle],config)+query(session,points[middle:],config)
    raise RuntimeError(f"Elevation batch failed (first location {points[0]}): {last}; cached earlier batches remain available")


def elevations(coordinates,config,cache_path,offline=False):
    if config.get("provider")=="dem_tiles":
        return tile_elevations(coordinates,config,offline)
    points=np.unique(np.asarray(coordinates,float),axis=0)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite elevation coordinates")
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    namespace=hashlib.sha256((config["service_url"]+"|bilinear|metres").encode()).hexdigest()
    key=lambda p:(int(round(p[0]*1e9)),int(round(p[1]*1e9)))
    with sqlite3.connect(cache_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS elevations (source TEXT, lon_key INTEGER, lat_key INTEGER, elevation REAL, resolution REAL, datum TEXT, fetched INTEGER, PRIMARY KEY(source,lon_key,lat_key))")
        cached={(x,y):(z,r,d,t) for x,y,z,r,d,t in connection.execute(
            "SELECT lon_key,lat_key,elevation,resolution,datum,fetched FROM elevations WHERE source=?",(namespace,))}
        missing=[point for point in points if key(point) not in cached]
        print(f"{len(points):,} unique points; {len(points)-len(missing):,} cached; {len(missing):,} to query",flush=True)
        if not offline:
            with requests.Session() as session:
                session.headers["User-Agent"]="IowaRWSCResearch-ElevationEnrichment/1.0"
                for start in range(0,len(missing),config["batch_size"]):
                    batch=missing[start:start+config["batch_size"]]
                    values=query(session,batch,config)
                    fetched=int(time.time())
                    records=[(namespace,*key(point),*value,fetched) for point,value in zip(batch,values)]
                    connection.executemany("INSERT OR REPLACE INTO elevations VALUES (?,?,?,?,?,?,?)",records)
                    connection.commit()
                    cached.update({key(point):(*value,fetched) for point,value in zip(batch,values)})
                    print(f"Elevation cache: {min(start+len(batch),len(missing)):,}/{len(missing):,} new points",flush=True)
        return [cached.get(key(point),(np.nan,np.nan,"unavailable",0)) for point in coordinates]


def bilinear_sample(dataset,x,y):
    data=dataset.read(1,masked=True)
    column,row=(~dataset.transform)*(np.asarray(x),np.asarray(y))
    column,row=np.asarray(column)-.5,np.asarray(row)-.5
    left,top=np.floor(column).astype(int),np.floor(row).astype(int)
    if (left<0).any() or (top<0).any() or (left+1>=dataset.width).any() or (top+1>=dataset.height).any():
        raise ValueError("Requested points outside padded DEM tile")
    dx,dy=column-left,row-top
    values=(data[top,left]*(1-dx)*(1-dy)+data[top,left+1]*dx*(1-dy)+
            data[top+1,left]*(1-dx)*dy+data[top+1,left+1]*dx*dy)
    return np.asarray(values.filled(np.nan),float)


def tile_elevations(coordinates,config,offline=False):
    import rasterio
    from pyproj import Transformer
    points=np.asarray(coordinates,float)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite elevation coordinates")
    transformer=Transformer.from_crs(4326,32615,always_xy=True)
    x,y=transformer.transform(points[:,0],points[:,1])
    resolution=float(config["dem_resolution_m"])
    width=float(config["dem_tile_width_m"])
    if resolution<=0 or width<=0 or not np.isclose(width/resolution,round(width/resolution)):
        raise ValueError("DEM tile width must be a positive multiple of pixel spacing")
    if width/resolution+4>8000:
        raise ValueError("DEM tile exceeds the service 8000-pixel limit; reduce tile width or increase pixel spacing")
    tile_ids=np.column_stack([np.floor(x/width),np.floor(y/width)]).astype(int)
    tiles=np.unique(tile_ids,axis=0)
    cache=resolve(config["dem_cache_directory"])
    cache.mkdir(parents=True,exist_ok=True)
    endpoint=config["service_url"].rsplit("/",1)[0]+"/exportImage"
    namespace=hashlib.sha256(f"{endpoint}|{resolution}|{width}|32615|bilinear|None".encode()).hexdigest()[:12]
    output=[(np.nan,resolution,"unavailable",0)]*len(points)
    print(f"Sampling {len(points):,} points from {len(tiles)} cached/downloadable {resolution:g}m DEM tiles",flush=True)
    with requests.Session() as session:
        for number,(tx,ty) in enumerate(tiles,1):
            path=cache/f"T{tx}_{ty}_{namespace}.tif"
            if not path.exists() and not offline:
                size=int(round(width/resolution))+4
                bounds=[tx*width-2*resolution,ty*width-2*resolution,
                        (tx+1)*width+2*resolution,(ty+1)*width+2*resolution]
                parameters={"f":"image","bbox":",".join(map(str,bounds)),"bboxSR":32615,"imageSR":32615,
                    "size":f"{size},{size}","format":"tiff","pixelType":"F32","compression":"LZ77",
                    "interpolation":"RSP_BilinearInterpolation","adjustAspectRatio":"false",
                    "renderingRule":json.dumps({"rasterFunction":"None"})}
                last=None
                for attempt in range(config["attempts"]):
                    temporary=path.with_suffix(".download.tmp")
                    try:
                        with session.get(endpoint,params=parameters,timeout=(15,config["read_timeout_seconds"]),stream=True) as response:
                            response.raise_for_status()
                            with open(temporary,"wb") as stream:
                                for chunk in response.iter_content(1024*1024):
                                    stream.write(chunk)
                        with rasterio.open(temporary) as source:
                            if source.count!=1 or source.dtypes[0]!="float32" or source.crs.to_epsg()!=32615:
                                raise ValueError("Expected georeferenced floating-point elevation, not a rendered image")
                            if not np.allclose(source.bounds,bounds,atol=.1,rtol=0):
                                raise ValueError("Downloaded DEM extent does not match the request")
                        temporary.replace(path)
                        break
                    except (requests.RequestException,ValueError,rasterio.errors.RasterioIOError) as error:
                        last=error
                        if attempt+1<config["attempts"]:
                            time.sleep(2**(attempt+1))
                else:
                    raise RuntimeError(f"DEM tile {tx},{ty} failed: {last}; previous tiles remain cached")
                time.sleep(config["request_gap_seconds"])
            if not path.exists():
                continue
            selected=np.flatnonzero((tile_ids[:,0]==tx)&(tile_ids[:,1]==ty))
            with rasterio.open(path) as dataset:
                values=bilinear_sample(dataset,np.asarray(x)[selected],np.asarray(y)[selected])
                if not np.allclose([dataset.res[0],dataset.res[1]],resolution,rtol=.01):
                    raise ValueError("Downloaded DEM pixel spacing differs from requested resolution")
            if not np.isfinite(values).all() or (values<-500).any() or (values>9000).any():
                raise ValueError(f"NoData/implausible DEM elevations in {path}")
            for position,value in zip(selected,values):
                output[int(position)]=(float(value),resolution,"USGS 3DEP CONUS / NAVD88",int(path.stat().st_mtime))
            print(f"DEM tile {number}/{len(tiles)}: {len(selected):,} points sampled",flush=True)
    return output


def reported_stations(stations,inventory):
    inventory=inventory.copy()
    inventory["NWS ID"]=inventory["NWS ID"].astype(str).str.strip()
    if inventory["NWS ID"].duplicated().any():
        raise ValueError("Duplicate NWS IDs in reported elevation inventory")
    lookup=inventory.set_index("NWS ID")
    output=stations.copy()
    output["reported_elevation_m"]=np.nan
    output["reported_station_name"]=""
    output["reported_match_method"]="not found"
    output["reported_matched_nws_id"]=""
    for i,row in output.iterrows():
        match=lookup.loc[row.rwis_id] if row.rwis_id in lookup.index else None
        method="NWS ID"
        if match is None:
            exact=inventory[np.isclose(inventory.Latitude,row.latitude,atol=1e-6,rtol=0)&
                            np.isclose(inventory["Longitude (-W)"],row.longitude,atol=1e-6,rtol=0)]
            if len(exact)==1:
                match=exact.iloc[0];method="identical coordinates; different NWS ID"
        if match is not None:
            output.loc[i,"reported_elevation_m"]=pd.to_numeric(match["Altitude (meters)"],errors="coerce")
            output.loc[i,"reported_station_name"]=str(match["RPU Name"])
            output.loc[i,"reported_match_method"]=method
            output.loc[i,"reported_matched_nws_id"]=str(match.name if method=="NWS ID" else match["NWS ID"])
    return output


def save_csv(frame,path):
    temporary=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(temporary,index=False,compression={"method":"gzip","compresslevel":4,"mtime":0} if path.suffix==".gz" else None)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"0c_elevation_config.json"))
    parser.add_argument("--max-support-points",type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    for key in ["batch_size","attempts","read_timeout_seconds"]:
        if config[key]<1:
            raise ValueError(f"{key} must be positive")
    if args.max_support_points is not None and args.max_support_points<1:
        raise ValueError("Positive support limit required")
    if args.max_support_points and not args.output_directory:
        raise ValueError("Reduced tests require a separate --output-directory")
    source=resolve(config["support_file"])
    roads=pd.read_csv(source)
    if roads.segment_id.duplicated().any() or roads.empty:
        raise ValueError("Expected unique support segment IDs")
    total=len(roads)
    if args.max_support_points:
        roads=roads.iloc[np.linspace(0,total-1,min(total,args.max_support_points),dtype=int)].copy()
    stations=pd.read_csv(resolve(config["station_file"]))
    inventory=pd.read_csv(resolve(config["reported_station_inventory"]),encoding="utf-8-sig")
    station_output=reported_stations(stations,inventory)
    output=resolve(args.output_directory or config["output_directory"])
    output.mkdir(parents=True,exist_ok=True)
    cache=resolve(config["cache_file"])
    # Stations first, so a completed station cache can already support fitting.
    station_values=elevations(stations[["longitude","latitude"]].to_numpy(),config,cache,args.offline)
    columns=["elevation_m","elevation_resolution_m","elevation_vertical_datum","elevation_retrieved_unix"]
    station_output[columns]=pd.DataFrame(station_values,index=station_output.index)
    station_output["elevation_source"]="USGS 3DEP DEM"
    station_output["dem_minus_reported_elevation_m"]=station_output.elevation_m-station_output.reported_elevation_m
    save_csv(station_output,output/"0c_rwis_stations_with_elevation.csv")
    values=elevations(roads[["longitude","latitude"]].to_numpy(),config,cache,args.offline)
    roads[columns]=pd.DataFrame(values,index=roads.index)
    roads["elevation_source"]="USGS 3DEP DEM"
    save_csv(roads,output/"0c_road_support_points_with_elevation.csv.gz")
    complete=bool(roads.elevation_m.notna().all() and station_output.elevation_m.notna().all())
    summary={"config":config,"original_support_count":total,"exported_support_count":len(roads),
        "partial_support_run":len(roads)!=total,"station_count":len(stations),"complete":complete,
        "missing_support_elevations":int(roads.elevation_m.isna().sum()),
        "missing_station_elevations":int(station_output.elevation_m.isna().sum()),
        "reported_station_elevations_available":int(station_output.reported_elevation_m.notna().sum()),
        "model_covariate":"elevation_m: common USGS terrain source for road points and RWIS; reported altitudes retained separately",
        "caveats":"DEM terrain elevation, not measured road-deck height. Reported GPS altitudes may use a different vertical reference. No station-only IDW elevation inference.",
        "script_sha256":fingerprint(Path(__file__)),
        "input_hashes":{str(path):fingerprint(path) for path in [source,resolve(config["station_file"]),resolve(config["reported_station_inventory"])]}}
    (output/"0c_elevation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Elevation outputs: {output}; complete={complete}",flush=True)
    if not complete:
        raise SystemExit("Missing elevations remain. Online rerun resumes the cache; do not fit with missing elevation values.")


if __name__=="__main__":
    main()
