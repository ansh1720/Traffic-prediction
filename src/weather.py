"""
Weather module.

Responsible for:
1. Fetching weather forecasts/observations from the Open-Meteo API.
2. Mapping WMO code to project-specific weather categories.
3. Estimating visibility and other values based on meteorological outputs.
4. Providing seasonal fallbacks when the API is down or outside forecast window.
"""

import urllib.request
import urllib.parse
import json
import math
import random
import datetime
import config

def get_weather_forecast(lat, lon, dt):
    """
    Get weather metrics for a specific coordinate and datetime.
    Returns a dict containing:
        - weather (str: Clear/Cloudy/Rain/Fog/Snow)
        - temperature (float: °C)
        - visibility_km (float: km)
        - wind_speed_kmh (float: km/h)
        - success (bool)
    """
    # Parse month for fallback calculations
    month = dt.month
    hour = dt.hour

    # Call Open-Meteo API (requires coordinates and optionally hourly params)
    try:
        # We query the forecast API. It provides 7 days forecast by default.
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "timezone": "auto"
        }
        url = f"{config.OPEN_METEO_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": config.API_USER_AGENT}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            
            if "hourly" in data:
                hourly = data["hourly"]
                times = hourly["time"]
                
                # Format target dt to match Open-Meteo format (YYYY-MM-DDTHH:00)
                target_str = dt.strftime("%Y-%m-%dT%H:00")
                
                # Find matching time index (or closest if not exact)
                idx = 0
                if target_str in times:
                    idx = times.index(target_str)
                else:
                    # Find closest time
                    min_diff = None
                    for i, t_str in enumerate(times):
                        try:
                            t_val = datetime.datetime.fromisoformat(t_str)
                            diff = abs((t_val - dt).total_seconds())
                            if min_diff is None or diff < min_diff:
                                min_diff = diff
                                idx = i
                        except:
                            pass
                
                # Extract values
                wmo_code = hourly["weather_code"][idx]
                temp = hourly["temperature_2m"][idx]
                wind_speed = hourly["wind_speed_10m"][idx]
                rh = hourly["relative_humidity_2m"][idx]
                
                # Map WMO code to project weather type
                weather = _map_wmo_code_to_weather(wmo_code)
                
                # Estimate visibility (not directly in standard Open-Meteo API)
                # Lower relative humidity means higher visibility, precip/fog lowers it
                visibility = _estimate_visibility(weather, rh)
                
                return {
                    "weather": weather,
                    "temperature": round(temp, 1),
                    "visibility_km": round(visibility, 1),
                    "wind_speed_kmh": round(wind_speed, 1),
                    "success": True
                }
    except Exception as e:
        print(f"Open-Meteo API error: {e}. Using seasonal fallback weather.")

    # Fallback to seasonal simulation matching our data generator patterns
    # Temperature varies by month (seasonal pattern)
    base_temp = 15.0 + 10.0 * math.sin(2.0 * math.pi * (month - 4.0) / 12.0)
    temperature = base_temp + random.normalvariate(0, 4.0)
    temperature = max(-10.0, min(45.0, temperature))

    # Weather probability distribution
    weather = random.choices(
        config.WEATHER_TYPES,
        weights=config.WEATHER_PROBS,
        k=1
    )[0]

    # Visibility base
    visibility_base = 10.0
    if weather == "Rain":
        visibility_base = 5.0
    elif weather == "Fog":
        visibility_base = 2.0
    elif weather == "Snow":
        visibility_base = 3.0
    elif weather == "Cloudy":
        visibility_base = 8.0
    visibility = visibility_base + random.normalvariate(0, 1.0)
    visibility = max(0.5, min(15.0, visibility))

    # Wind speed
    wind_speed = random.expovariate(1.0 / 10.0)
    wind_speed = max(0.0, min(60.0, wind_speed))

    return {
        "weather": weather,
        "temperature": round(temperature, 1),
        "visibility_km": round(visibility, 1),
        "wind_speed_kmh": round(wind_speed, 1),
        "success": False
    }


def _map_wmo_code_to_weather(wmo_code):
    """Map WMO weather codes to our five categories: Clear, Cloudy, Rain, Fog, Snow."""
    if wmo_code in [0]:
        return "Clear"
    elif wmo_code in [1, 2, 3]:
        return "Cloudy"
    elif wmo_code in [45, 48]:
        return "Fog"
    elif wmo_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]:
        return "Rain"
    elif wmo_code in [71, 73, 75, 77, 85, 86]:
        return "Snow"
    else:
        return "Clear"


def _estimate_visibility(weather, relative_humidity):
    """Estimate visibility in km based on weather type and humidity."""
    if weather == "Fog":
        return 1.5 + random.uniform(-0.5, 0.5)
    elif weather == "Snow":
        return 3.0 + random.uniform(-1.0, 1.0)
    elif weather == "Rain":
        return 5.0 + random.uniform(-1.5, 1.5)
    elif weather == "Cloudy":
        # Higher humidity slightly lowers visibility
        factor = (100.0 - relative_humidity) / 100.0
        return 7.0 + 3.0 * factor
    else: # Clear
        factor = (100.0 - relative_humidity) / 100.0
        return 11.0 + 4.0 * factor
