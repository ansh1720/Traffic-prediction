"""
Traffic Congestion Prediction & Travel Planner — Lightweight Flask Server.

Serves pure HTML/CSS/JS frontend, exposes REST APIs for geocoding, OSRM routing,
Open-Meteo weather forecasts, historical segment feature lookup, and model serving.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import joblib
import urllib.request
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.predictor import predict_single, predict_batch, load_model, load_scaler, load_label_encoder, lookup_features_by_coord
from src.routing import geocode_address, get_route
from src.weather import get_weather_forecast

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# Load pre-trained models and preprocessing objects once at startup
MODEL = load_model()
SCALER = load_scaler()
LABEL_ENCODER = load_label_encoder()
FEATURE_NAMES = joblib.load(os.path.join(config.MODELS_DIR, "feature_names.pkl"))


def get_alternative_route(start_coords, end_coords):
    """Retrieve alternative driving route from OSRM engine."""
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords
    url = f"{config.OSRM_API_URL}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson&steps=true&alternatives=true"
    
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": config.API_USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("code") == "Ok" and len(data.get("routes", [])) > 1:
                route = data["routes"][1]
                distance_km = route["distance"] / 1000.0
                duration_mins = route["duration"] / 60.0
                geojson_coords = route["geometry"]["coordinates"]
                geometry = [[c[1], c[0]] for c in geojson_coords]
                
                # Parse steps
                segments = []
                steps = route.get("legs", [{}])[0].get("steps", [])
                for step in steps:
                    name = step.get("name", "")
                    ref = step.get("ref", "")
                    step_dist = step.get("distance", 0) / 1000.0
                    if step_dist < 0.01:
                        continue
                    
                    loc = step.get("maneuver", {}).get("location", [lon1, lat1])
                    coord = [loc[1], loc[0]]
                    
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
                
                return {
                    "success": True,
                    "distance_km": round(distance_km, 2),
                    "duration_mins": round(duration_mins, 1),
                    "geometry": geometry,
                    "segments": segments
                }
    except Exception as e:
        pass
    return None


def predict_route(route_data, weather_data, dt, incident_nearby):
    """Predict traffic levels and adjust durations for OSRM segments."""
    total_nominal_duration = 0.0
    total_predicted_duration = 0.0
    segment_details = []
    
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if hour in config.PEAK_HOURS else 0
    is_holiday = 0
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week_name = day_names[day_of_week]
    
    if hour <= 5 or hour > 20:
        time_bucket = "Night"
    elif hour <= 11:
        time_bucket = "Morning"
    elif hour <= 16:
        time_bucket = "Afternoon"
    else:
        time_bucket = "Evening"
        
    for seg in route_data["segments"]:
        road_type = seg["road_type"]
        dist = seg["distance_km"]
        name = seg["name"]
        coord = seg.get("coord", [12.9716, 77.5946])
        
        # Spatial lookup from historical cache
        static_feats = lookup_features_by_coord(coord[0], coord[1])
        
        speed_limit = config.ROAD_SPEED_LIMITS.get(road_type, 40.0)
        nominal_dur = (dist / speed_limit) * 60.0
        
        feature_dict = {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "is_holiday": is_holiday,
            "Weather": weather_data["weather"],
            "Temperature": weather_data["temperature"],
            "RoadType": static_feats.get("RoadType", road_type),
            "LargeVehicles": static_feats.get("LargeVehicles", "Allowed"),
            "Landmarks": static_feats.get("Landmarks", "No"),
            "NumberofLanes": static_feats.get("NumberofLanes", 2),
            "incident_nearby": int(incident_nearby),
            "day_of_week_name": day_of_week_name,
            "time_of_day_bucket": time_bucket
        }
        
        # Serving prediction
        pred_res = predict_single(feature_dict, MODEL, SCALER, LABEL_ENCODER, FEATURE_NAMES)
        pred_level = pred_res["prediction"]
        confidence = pred_res["confidence"]
        
        # Adjust travel time based on congestion multiplier
        multiplier = config.CONGESTION_TIME_MULTIPLIERS.get(pred_level, 1.0)
        predicted_dur = nominal_dur * multiplier
        
        total_nominal_duration += nominal_dur
        total_predicted_duration += predicted_dur
        
        segment_details.append({
            "name": name,
            "road_type": static_feats.get("RoadType", road_type),
            "distance_km": round(dist, 2),
            "nominal_time_mins": round(nominal_dur, 1),
            "predicted_congestion": pred_level,
            "predicted_time_mins": round(predicted_dur, 1),
            "confidence": round(confidence * 100, 1),
            "lanes": static_feats.get("NumberofLanes", 2),
            "large_vehicles": static_feats.get("LargeVehicles", "Allowed")
        })
        
    return total_nominal_duration, total_predicted_duration, segment_details


@app.route("/")
def index():
    """Serve lightweight HTML UI template."""
    return render_template("index.html")


@app.route("/api/route", methods=["POST"])
def api_route():
    """Predict traffic for a routed journey between origin and destination."""
    data = request.json or {}
    src = data.get("source", "").strip()
    dst = data.get("destination", "").strip()
    date_str = data.get("date", "")
    time_str = data.get("time", "")
    incident = data.get("incident", False)
    
    if not src or not dst:
        return jsonify({"success": False, "error": "Source and destination are required"}), 400
        
    start_coords = geocode_address(src)
    end_coords = geocode_address(dst)
    
    if not start_coords or not end_coords:
        return jsonify({"success": False, "error": "Failed to geocode locations"}), 400
        
    # Query primary route
    route_data = get_route(start_coords, end_coords)
    if not route_data or not route_data.get("geometry"):
        return jsonify({"success": False, "error": "Failed to fetch routing path"}), 400
        
    # Query alternative route
    alt_route = get_alternative_route(start_coords, end_coords)
    
    try:
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except Exception:
        dt = datetime.datetime.now()
        
    weather_data = get_weather_forecast(start_coords[0], start_coords[1], dt)
    
    nom_dur, pred_dur, segments = predict_route(route_data, weather_data, dt, incident)
    delay = max(0.0, pred_dur - nom_dur)
    
    # Generate explanations
    explanations = []
    hour = dt.hour
    is_peak = hour in config.PEAK_HOURS
    is_weekend = dt.weekday() >= 5
    
    if is_peak:
        explanations.append("<b>Peak Commuter Rush:</b> Overlaps with morning/evening rush hours, causing higher density.")
    else:
        explanations.append("<b>Off-Peak Traffic:</b> Outside core commuting windows.")
        
    if is_weekend:
        explanations.append("<b>Weekend Factor:</b> Lower general density, but localized peaks may occur.")
    else:
        explanations.append("<b>Weekday Volume:</b> Standard corporate/school schedules are active.")

    if config.WEATHER_SEVERITY.get(weather_data["weather"], 0) >= 2:
        explanations.append(f"<b>Weather Impact ({weather_data['weather']}):</b> Adverse weather slows driving speeds and increases delays.")
        
    if incident:
        explanations.append("<b>Local Incident:</b> Active construction or collision reported nearby.")
        
    high_segs = [s for s in segments if s["predicted_congestion"] == "High"]
    if high_segs:
        explanations.append(f"<b>Bottlenecks:</b> High congestion predicted on segments like '{high_segs[0]['name']}'.")
        
    return jsonify({
        "success": True,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "weather": weather_data,
        "distance_km": route_data["distance_km"],
        "nominal_time": round(nom_dur, 1),
        "predicted_time": round(pred_dur, 1),
        "delay": round(delay, 1),
        "segments": segments,
        "route_geometry": route_data["geometry"],
        "alt_route": alt_route,
        "explanations": explanations,
        "route_data": route_data  # Pass back to front for optimizer reuse
    })


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """Predict travel times across offset windows for the departure chart."""
    data = request.json or {}
    route_data = data.get("route_data")
    weather_data = data.get("weather_data")
    date_str = data.get("date")
    time_str = data.get("time")
    incident = data.get("incident", False)
    
    if not route_data or not weather_data:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
        
    try:
        base_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except Exception:
        base_dt = datetime.datetime.now()
        
    times = []
    durations = []
    
    # Calculate travel times in a offset loop (-2 hours to +3 hours)
    for offset in range(-2, 4):
        offset_dt = base_dt + datetime.timedelta(hours=offset)
        _, test_dur, _ = predict_route(route_data, weather_data, offset_dt, incident)
        times.append(offset_dt.strftime("%I:%M %p"))
        durations.append(round(test_dur, 1))
        
    return jsonify({
        "times": times,
        "durations": durations
    })


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Process a batch of traffic records uploaded as CSV."""
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400
        
    try:
        raw_df = pd.read_csv(file)
        res_df = predict_batch(raw_df, MODEL, SCALER, LABEL_ENCODER, FEATURE_NAMES)
        
        # Save output in project scratch dir
        temp_path = os.path.join(config.PROJECT_ROOT, "scratch", "predictions_output.csv")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        res_df.to_csv(temp_path, index=False)
        return send_file(temp_path, as_attachment=True, download_name="predictions_output.csv")
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    print("--------------------------------------------------")
    print(" Lightweight HTML/CSS/JS Travel Planner Server Active ")
    print(" Local URL: http://127.0.0.1:5000                     ")
    print("--------------------------------------------------")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
