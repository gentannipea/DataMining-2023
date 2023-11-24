import geopandas as gpd
from shapely.geometry import Point

def get_state_from_coordinates(lat, lon):
    # Load the GeoJSON file containing state boundaries
    states = gpd.read_file("states.geojson")

    # Create a GeoDataFrame for the point
    geometry = [Point(lon, lat)]
    point_gdf = gpd.GeoDataFrame(geometry, columns=['geometry'], crs=states.crs)

    # Use spatial join to find the state
    state_data = gpd.sjoin(states, point_gdf, op="contains")
    if not state_data.empty:
        state_name = state_data.iloc[0]['NAME']
        return state_name
    else:
        return "Not Found"

# Example usage:
latitude = 40.7128  # Replace with your latitude
longitude = -74.0060  # Replace with your longitude

state = get_state_from_coordinates(latitude, longitude)
if state:
    print(f"The coordinates ({latitude}, {longitude}) are in the state of {state}")
else:
    print("The state could not be determined for the given coordinates.")
