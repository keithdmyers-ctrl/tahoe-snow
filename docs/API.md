# API Reference

Flask web app served by `webapp.py`. Default port: 5000.

## Endpoints

### GET /
Serves the web dashboard (single-page application).

### GET /api/data
Returns the full analysis JSON. Cached for 15 minutes.

**Response**: JSON object containing:
- `generated`: ISO timestamp of when data was generated
- `current_conditions`: Current NWS observation + lake-level conditions
- `resorts`: Per-resort forecast data (Heavenly, Northstar, Kirkwood)
  - Each resort has `zones` (base, mid, peak) with `current`, `timeline_48h`, `day_night_buckets`, `model_spread`
- `snotel_current`, `snotel_history`, `season_stats`: Snowpack data
- `avalanche`: Sierra Avalanche Center danger rating
- `chains`: Caltrans chain control status
- `lifts`: Per-resort lift open/total counts
- `decision`: Ski decision score and reasoning
- `storm_narrative`: Meteorologist-style briefing
- `storm_history`: Archived past storms
- `hero_stats`: Key numbers for dashboard hero cards
- `summary`: One-line text summary
- `model_weights`: Current BMA model weights
- `ensemble`, `nbm`, `nws_grids`, `normals`, `synoptic`, `rwis`: Additional data sources

### GET /api/verification
Returns forecast verification summary.

**Response**: JSON object with per-model MAE/RMSE/bias, model weights, PoP calibration, skill score.

### GET /api/decision
Returns just the ski decision data.

**Response**: `{ decision: {...}, storm_narrative: "...", storm_history: [...] }`

### GET /api/refresh
Force data refresh. Throttled to once per 60 seconds.

**Response**: `{ status: "ok", generated: "..." }` or `{ status: "throttled", retry_after: N }` (429)

## Cache Behavior
- Default TTL: 15 minutes (900 seconds)
- Refresh throttle: 60 seconds minimum between forced refreshes
- Background pre-fetch on server startup
