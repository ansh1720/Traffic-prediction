"""
Routing module.

Responsible for:
1. Geocoding addresses using Nominatim API.
2. Finding driving routes and geometry using OSRM API.
3. Extracting route features and segmenting the route by road types.
4. Providing fallback behavior if the APIs are down or rate-limited.
5. Mapping Bengaluru coordinates to the dataset's geohash coordinate space.
"""

import urllib.request
import urllib.parse
import json
import re
import math
import random
import config

def geocode_address(address):
    """
    Geocode an address string to (latitude, longitude) using Nominatim.
    Returns None on failure — never uses random/hash fallback coordinates.
    """
    if not address or not address.strip():
        return None

    address_clean = address.strip()

    # Pre-defined mocks for common Bengaluru landmarks (fast path, no API call needed)
    mocks = {
        "majestic": (12.9781, 77.5696),
        "kempegowda bus": (12.9781, 77.5696),
        "koramangala": (12.9352, 77.6244),
        "indiranagar": (12.9719, 77.6412),
        "whitefield": (12.9698, 77.7500),
        "electronic city": (12.8452, 77.6602),
        "airport": (13.1986, 77.7066),
        "kempegowda airport": (13.1986, 77.7066),
        "manyata": (13.0451, 77.6266),
        "mg road": (12.9740, 77.6085),
        "shanti nagar": (12.9575, 77.5957),
        "wilson garden": (12.9500, 77.5950),
        "jayanagar": (12.9308, 77.5831),
        "hsr layout": (12.9081, 77.6476),
        "btm layout": (12.9166, 77.6101),
        "banashankari": (12.9256, 77.5468),
        "hebbal": (13.0352, 77.5970),
        "yelahanka": (13.1007, 77.5963),
        "marathahalli": (12.9591, 77.6974),
        "jp nagar": (12.9102, 77.5765),
        "rajajinagar": (12.9915, 77.5521),
        "malleshwaram": (13.0035, 77.5710),
        "basavanagudi": (12.9430, 77.5748),
        "lalbagh": (12.9507, 77.5848),
    }

    for key, coords in mocks.items():
        if key in address_clean.lower():
            return coords

    def _call_nominatim(query, bounded=False):
        """Make a Nominatim API call, optionally bounded to Bengaluru."""
        params = {"q": query, "format": "json", "limit": 1, "countrycodes": "in"}
        if bounded:
            params["viewbox"] = "77.45,13.10,77.75,12.85"
            params["bounded"] = 1
        url = f"{config.NOMINATIM_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": config.API_USER_AGENT})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
        return None

    # Attempt 1: Full address, bounded to Bengaluru
    try:
        result = _call_nominatim(address_clean, bounded=True)
        if result:
            return result
    except Exception as e:
        print(f"Geocoding attempt 1 failed: {e}")

    # Attempt 2: If it looks like a Nominatim display_name (has commas),
    # use only the first part (place name) + Bengaluru
    if "," in address_clean:
        short_name = address_clean.split(",")[0].strip()
        if short_name and len(short_name) > 2:
            try:
                result = _call_nominatim(f"{short_name}, Bengaluru", bounded=True)
                if result:
                    print(f"Geocoding succeeded with short name: '{short_name}'")
                    return result
            except Exception as e:
                print(f"Geocoding attempt 2 failed: {e}")

    # All attempts failed — return None so caller shows an error
    print(f"Geocoding completely failed for: '{address_clean[:80]}'")
    return None


def get_route(start_coords, end_coords):
    """
    Query OSRM API to get route details between start_coords and end_coords.
    Returns a dict with:
        - success (bool)
        - distance_km (float)
        - duration_mins (float)
        - geometry (list of [lat, lon])
        - segments (list of dicts containing distance and estimated road type)
    """
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords

    url = f"{config.OSRM_API_URL}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson&steps=true"
    
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": config.API_USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                distance_km = route["distance"] / 1000.0
                duration_mins = route["duration"] / 60.0
                
                # Extract coordinates
                geojson_coords = route["geometry"]["coordinates"]
                geometry = [[c[1], c[0]] for c in geojson_coords] # OSRM returns lon, lat -> convert to lat, lon
                
                # Extract road segments from steps
                segments = []
                steps = route.get("legs", [{}])[0].get("steps", [])
                
                for step in steps:
                    name = step.get("name", "")
                    ref = step.get("ref", "")
                    step_dist = step.get("distance", 0) / 1000.0
                    
                    if step_dist < 0.01:
                        continue
                        
                    # Extract starting coordinate for this step
                    loc = step.get("maneuver", {}).get("location", [lon1, lat1])
                    coord = [loc[1], loc[0]] # [lat, lon]
                    
                    # Classify road type
                    road_type = "Local"
                    combined_name = f"{name} {ref}".lower()
                    if any(x in combined_name for x in ["expressway", "freeway", "highway", "i-", "us-", "interstate", "parkway"]):
                        road_type = "Highway"
                    elif any(x in combined_name for x in ["avenue", "boulevard", "broadway", "rt-", "route"]):
                        road_type = "Arterial"
                    elif any(x in combined_name for x in ["street", "drive", "road", "way"]):
                        road_type = "Collector"
                        
                    segments.append({
                        "road_type": road_type,
                        "distance_km": step_dist,
                        "name": name if name else "Unnamed road",
                        "coord": coord
                    })
                
                # If OSRM returns no steps, build simple proportions
                if not segments:
                    segments = _generate_proportional_segments(distance_km, start_coords, end_coords)
                    
                return {
                    "success": True,
                    "distance_km": round(distance_km, 2),
                    "duration_mins": round(duration_mins, 1),
                    "geometry": geometry,
                    "segments": segments
                }
    except Exception as e:
        print(f"OSRM Routing API error: {e}. Using simulated routing fallback.")

    # Fallback simulation
    return _generate_fallback_route(start_coords, end_coords)


def _generate_proportional_segments(distance_km, start_coords=None, end_coords=None):
    """Generate segments proportionally based on total distance when steps are unavailable."""
    if start_coords is None:
        start_coords = (12.9716, 77.5946)
    if end_coords is None:
        end_coords = (12.9716, 77.5946)
    
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords

    if distance_km > 15:
        # High highway share for long trips
        return [
            {"road_type": "Local", "distance_km": round(distance_km * 0.1, 2), "name": "Local Streets", "coord": [lat1, lon1]},
            {"road_type": "Arterial", "distance_km": round(distance_km * 0.2, 2), "name": "Arterial Roads", "coord": [lat1 + (lat2-lat1)*0.2, lon1 + (lon2-lon1)*0.2]},
            {"road_type": "Highway", "distance_km": round(distance_km * 0.6, 2), "name": "Expressway", "coord": [lat1 + (lat2-lat1)*0.5, lon1 + (lon2-lon1)*0.5]},
            {"road_type": "Collector", "distance_km": round(distance_km * 0.1, 2), "name": "Collector Road", "coord": [lat2, lon2]}
        ]
    else:
        # Mostly local and arterials for short trips
        return [
            {"road_type": "Local", "distance_km": round(distance_km * 0.3, 2), "name": "Local Streets", "coord": [lat1, lon1]},
            {"road_type": "Collector", "distance_km": round(distance_km * 0.3, 2), "name": "Collector Street", "coord": [lat1 + (lat2-lat1)*0.4, lon1 + (lon2-lon1)*0.4]},
            {"road_type": "Arterial", "distance_km": round(distance_km * 0.4, 2), "name": "Arterial Road", "coord": [lat2, lon2]}
        ]


def _generate_fallback_route(start_coords, end_coords):
    """Generate a straight line path with some intermediate points and simulated segment features."""
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords

    # Haversine formula to compute great circle distance
    R = 6371.0 # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c

    # Estimate duration based on average speeds (approx 50 km/h overall)
    duration_mins = (distance_km / 50.0) * 60.0

    # Interpolate points for the geometry
    num_points = max(5, int(distance_km * 2))
    num_points = min(num_points, 50)
    
    geometry = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        lat = lat1 + (lat2 - lat1) * fraction
        lon = lon1 + (lon2 - lon1) * fraction
        # Add a tiny wiggle to look realistic on map
        if i > 0 and i < num_points - 1:
            lat += random.uniform(-0.001, 0.001)
            lon += random.uniform(-0.001, 0.001)
        geometry.append([lat, lon])

    segments = _generate_proportional_segments(distance_km, start_coords, end_coords)

    return {
        "success": False, # Flagged as fallback
        "distance_km": round(distance_km, 2),
        "duration_mins": round(duration_mins, 1),
        "geometry": geometry,
        "segments": segments
    }


def translate_blr_to_dataset(lat, lon):
    """
    Linearly scale and translate Bengaluru coordinates to the dataset coordinate space.
    Bengaluru bounds: lat [12.85, 13.10], lon [77.45, 77.75]
    Dataset bounds: lat [-5.485, -5.237], lon [90.587, 90.972]
    """
    # Clamp inputs to Bengaluru box
    lat_clamped = max(config.BENGALURU_LAT_MIN, min(config.BENGALURU_LAT_MAX, lat))
    lon_clamped = max(config.BENGALURU_LON_MIN, min(config.BENGALURU_LON_MAX, lon))

    # Calculate proportions (0.0 to 1.0)
    lat_pct = (lat_clamped - config.BENGALURU_LAT_MIN) / (config.BENGALURU_LAT_MAX - config.BENGALURU_LAT_MIN)
    lon_pct = (lon_clamped - config.BENGALURU_LON_MIN) / (config.BENGALURU_LON_MAX - config.BENGALURU_LON_MIN)

    # Map to dataset coordinate range
    lat_ds = config.DATASET_LAT_MIN + lat_pct * (config.DATASET_LAT_MAX - config.DATASET_LAT_MIN)
    lon_ds = config.DATASET_LON_MIN + lon_pct * (config.DATASET_LON_MAX - config.DATASET_LON_MIN)

    return lat_ds, lon_ds


def encode_geohash(latitude, longitude, precision=6):
    """
    Encode coordinates to geohash string.
    Standard dependency-free geohash encoder.
    """
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(base32[ch])
            bit = 0
            ch = 0
    return "".join(geohash)


def get_dataset_geohash(lat, lon):
    """
    Translates Bengaluru coordinates to dataset coordinate space and encodes as geohash.
    """
    lat_ds, lon_ds = translate_blr_to_dataset(lat, lon)
    return encode_geohash(lat_ds, lon_ds, precision=6)
