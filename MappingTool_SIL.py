import os
import io
import base64
import pickle
import requests
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import folium
from folium.plugins import FeatureGroupSubGroup, MarkerCluster
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon, MultiPolygon, shape, Point
from shapely.strtree import STRtree
import shapefile
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import lang2vec.lang2vec as l2v
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
geolocator = Nominatim(user_agent="lang_mapper")

from ollama import chat, ChatResponse

def fetch_barriers_to_entry(lang_name, country):
    try:
        response: ChatResponse = chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': f'What are the barriers to entry for the {lang_name} language in {country}? Limit response to 3 short bullet points. Base the response on an accredited website, like Harvard Business Review. Also, if available, please provide the Government restriction score as well. Please ensure you do not say anything like "This language does not exist". If you are unable to find information for the language give general details related to the country or closest city. Please share only one value for government restriction score. '
            },
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error fetching LLM barriers: {e}")
        return None

def load_data():
    print("Loading language data...")
    file_path = "ProgressBible Data with Coordinates.xlsx" #Enter file name here
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name='Language List', dtype=str)
    df.replace("#N/A", np.nan, inplace=True)
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype(float)
    
    # Exclude invalid lat/lon, including near (0,0)
    df = df[
        (df['Latitude'].between(-90, 90)) &
        (df['Longitude'].between(-180, 180)) &
        ~((df['Latitude'].abs() < 1e-3) & (df['Longitude'].abs() < 1e-3))
    ]

    df = df[~df['LanguageName'].str.contains('Sign Language', case=False, na=False)]

    hub_languages = df[df['Scripture'].notna()].copy()
    untranslated_languages = df[df['Scripture'].isna()].copy()
    hub_coords = hub_languages[['Language Code', 'Latitude', 'Longitude']].dropna().drop_duplicates()
    untranslated_coords = untranslated_languages[['Language Code', 'Latitude', 'Longitude']].dropna()

    hub_locations = hub_coords[['Latitude', 'Longitude']].values
    print(f"Loaded {len(df)} total languages, {len(hub_coords)} hubs, and {len(untranslated_coords)} untranslated.")
    return df, hub_coords, untranslated_coords, hub_locations

def load_churches():
    print("Loading church data...")
    church_df = pd.read_csv("churches_by_country.csv")
    church_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    church_df[['Latitude', 'Longitude']] = church_df[['Latitude', 'Longitude']].astype(float)
    print(f"Loaded {len(church_df)} churches.")
    return church_df

def load_shapefile():
    print("Loading shapefile (land mask)...")
    sf = shapefile.Reader("ne_10m_land.shp")
    land_polygons = []
    for record in sf.shapeRecords():
        geom = shape(record.shape.__geo_interface__)
        if geom.geom_type == "Polygon":
            land_polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            land_polygons.extend(geom.geoms)
    print("Shapefile loaded.")
    return MultiPolygon(land_polygons)

def compute_voronoi(hub_locations, land_mask, hub_coords, df):
    hub_locations_xy = np.array([[lon, lat] for lat, lon in hub_locations])
    print("Generating Voronoi diagram...")
    vor = Voronoi(hub_locations_xy)
    voronoi_polygons = []
    all_points = df[['Language Code', 'Latitude', 'Longitude', 'Scripture']].dropna(subset=['Latitude', 'Longitude'])
    all_points[['Latitude', 'Longitude']] = all_points[['Latitude', 'Longitude']].astype(float)
    all_points['geometry'] = [Point(xy) for xy in zip(all_points['Longitude'], all_points['Latitude'])]
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue
        vertices = [vor.vertices[j] for j in region]
        if len(vertices) < 3:
            continue
        polygon = Polygon(vertices).intersection(land_mask)
        if polygon.is_empty:
            continue

        # Step 1: Get languages covered by polygon
        languages_in_region = all_points[[polygon.covers(p) for p in all_points['geometry']]]

        # Step 2: Ensure hub language is included
        hub_lang_code = hub_coords.iloc[i]['Language Code']
        hub_lat = hub_coords.iloc[i]['Latitude']
        hub_lon = hub_coords.iloc[i]['Longitude']
        hub_point = Point(hub_lon, hub_lat)

        # Step 3: Include the hub language if missing
        if not polygon.covers(hub_point):
            print(f"Hub point for {hub_lang_code} not covered by its own Voronoi polygon (Region #{i})")

        # Even if it’s not detected, add it if it’s in the dataframe
        if hub_lang_code not in languages_in_region['Language Code'].values:
            hub_row = df[df['Language Code'] == hub_lang_code]
            if not hub_row.empty:
                print(f"Adding hub '{hub_lang_code}' manually to region #{i}")
                languages_in_region = pd.concat([languages_in_region, hub_row], ignore_index=True)

        translated = languages_in_region[languages_in_region['Scripture'].notna()]['Language Code'].tolist()
        untranslated = languages_in_region[languages_in_region['Scripture'].isna()]['Language Code'].tolist()
        translation_ratio = len(translated) / len(languages_in_region) if len(languages_in_region) > 0 else 0
        voronoi_polygons.append((
            hub_coords.iloc[i]['Language Code'],
            polygon,
            translation_ratio,
            len(translated),
            len(untranslated),
            translated,
            untranslated
        ))
    print("Finished computing Voronoi regions.")
    print("Tracking skipped hubs")
    
    return voronoi_polygons


def get_coordinates_from_city(city_name):
    try:
        location = geolocator.geocode(city_name)
        return (location.latitude, location.longitude) if location else None
    except:
        return None

def get_nearest_language_code(lat, lon):
    _, index = cKDTree(df[['Latitude', 'Longitude']].values).query([lat, lon])
    return df.iloc[index]['Language Code']

def generate_country_pie_chart(country_name):
    country_df = df[df['Country'] == country_name]
    translated = country_df[country_df['Scripture'].notna()]
    untranslated = country_df[country_df['Scripture'].isna()]
    counts = [len(translated), len(untranslated)]
    fig, ax = plt.subplots()
    ax.pie(counts, labels=["Translated", "Untranslated"], autopct="%1.1f%%", startangle=90, colors=["green", "red"])
    ax.axis('equal')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return encoded_image

def get_travel_time(start, end, mode='driving'):
    try:
        base_url = f"http://router.project-osrm.org/route/v1/{mode}/"
        coordinates = f"{start[1]},{start[0]};{end[1]},{end[0]}"
        url = f"{base_url}{coordinates}?overview=false"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            routes = data.get('routes', [])
            if routes:
                duration = routes[0]['duration']
                return round(duration / 60, 2)
    except Exception as e:
        print(f"Travel time API error: {e}")
    return None


def generate_map(input_lat, input_lon, input_language_code, closest_hub, hub_distance_km, hub_travel_time, closest_church, church_distance_km, church_travel_time, voronoi_polygons, church_df, similarity_score):
    folium_map = folium.Map(location=[input_lat, input_lon], zoom_start=4)

    for lang_code, polygon, translation_ratio, num_translated, num_untranslated, translated, untranslated in voronoi_polygons:
        color = "green" if translation_ratio > 0.8 else "red" if translation_ratio < 0.2 else "yellow"
        popup_html = (
            f"<b>Hub Language:</b> {lang_code}<br>"
            f"<b>Translated Languages:</b> {num_translated}<br>"
            f"<b>Untranslated Languages:</b> {num_untranslated}<br>"
            f"<b>Translation Ratio:</b> {translation_ratio:.2f}"
        )
        folium.GeoJson(
            polygon.__geo_interface__,
            style_function=lambda x, col=color: {
                "fillColor": col, "color": "black", "weight": 1, "fillOpacity": 0.4
            },
            tooltip=popup_html
        ).add_to(folium_map)

    marker_cluster = MarkerCluster().add_to(folium_map)
    translated_sub = FeatureGroupSubGroup(marker_cluster, 'Translated Languages')
    untranslated_sub = FeatureGroupSubGroup(marker_cluster, 'Untranslated Languages')
    folium_map.add_child(translated_sub)
    folium_map.add_child(untranslated_sub)

    for _, row in hub_coords.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Hub Language: {row['Language Code']}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(translated_sub)

    for _, row in untranslated_coords.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Untranslated Language: {row['Language Code']}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(untranslated_sub)

    church_group = folium.FeatureGroup(name='Churches', show=True)
    for _, row in church_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            popup=f"Church: {row['Name']}, {row['Country']}",
            tooltip=row['Name'],
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=0.7
        ).add_to(church_group)
    folium_map.add_child(church_group)

    folium.Marker(
        location=[input_lat, input_lon],
        popup=(f"<b>Input Language:</b> {input_language_code}<br>"
               f"<b>Similarity Score:</b> {similarity_score}<br>"
               f"<b>Closest Hub:</b> {closest_hub['Language Code']} ({hub_distance_km} km, {hub_travel_time} min)<br>"
               f"<b>Closest Church:</b> {closest_church['Name']} ({church_distance_km} km, {church_travel_time} min)"),
        icon=folium.Icon(color="green", icon="flag")
    ).add_to(folium_map)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: auto;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px; border-radius: 8px;">
        <b>Legend</b><br>
        <i style="background:green; width:10px; height:10px; display:inline-block; margin-right:5px;"></i>High Translation Ratio (>80%)<br>
        <i style="background:yellow; width:10px; height:10px; display:inline-block; margin-right:5px;"></i>Moderate Translation Ratio (20%-80%)<br>
        <i style="background:red; width:10px; height:10px; display:inline-block; margin-right:5px;"></i>Low Translation Ratio (<20%)<br>
    </div>
    '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))

    return folium_map


def get_language_similarity(lang1, lang2, features=None):
    if features is None:
        features = ["syntax_knn", "phonology_average", "inventory_average", "fam"]
    valid_features = l2v.available_feature_sets()
    for f in features:
        if f not in valid_features:
            raise ValueError(f"Feature set '{f}' is not available.")
    vec1_all, vec2_all = [], []
    for feature in features:
        data = l2v.get_features([lang1, lang2], feature)
        vec1, vec2 = data.get(lang1), data.get(lang2)
        if vec1 is None or vec2 is None:
            continue
        filtered = [(float(x), float(y)) for x, y in zip(vec1, vec2) if x != '--' and y != '--']
        if filtered:
            v1_filtered, v2_filtered = zip(*filtered)
            vec1_all.extend(v1_filtered)
            vec2_all.extend(v2_filtered)
    if not vec1_all or not vec2_all:
        raise RuntimeError("No usable features found for comparison.")
    return round(cosine_similarity(np.array(vec1_all).reshape(1, -1), np.array(vec2_all).reshape(1, -1))[0][0], 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form.get('input_type')
        input_value = request.form.get('input_value').strip()
        barriers_info = None
        country_image = None
        similarity_score = None
        if input_type == 'language':
            input_language = df[df['Language Code'] == input_value]
            if input_language.empty:
                return render_template('index6.html', error=f"Language code '{input_value}' not found.")
            input_lat, input_lon = input_language.iloc[0][['Latitude', 'Longitude']]
            lang_row = input_language.iloc[0]
            print("Fetching barriers to entry from LLM...")
            barriers_info = fetch_barriers_to_entry(lang_row['LanguageName'], lang_row['Country'])

        elif input_type == 'city':
            coords = get_coordinates_from_city(input_value)
            if not coords:
                return render_template('index6.html', error=f"City '{input_value}' not found.")
            input_lat, input_lon = coords
            input_value = get_nearest_language_code(input_lat, input_lon)
            lang_row = df[df['Language Code'] == input_value].iloc[0]
            print("Fetching barriers to entry from LLM...")
            barriers_info = fetch_barriers_to_entry(lang_row['LanguageName'], lang_row['Country'])

        elif input_type == 'country':
            if input_value not in df['Country'].dropna().unique():
                return render_template('index6.html', error=f"Country '{input_value}' not found.")
            country_image = generate_country_pie_chart(input_value)
            return render_template("index6.html", country_image=country_image, map_path="static/voronoi_language_map.html")

        else:
            return render_template('index6.html', error="Invalid input type.")

        _, hub_index = cKDTree(hub_locations).query([input_lat, input_lon])
        print("Finding nearest hub language...")
        closest_hub = hub_coords.iloc[hub_index]
        print("Finding similarity score...")
        similarity_score = get_language_similarity(input_value, closest_hub['Language Code'])
        print("Finding travel distance and travel time...")
        hub_distance_km = round(geodesic((input_lat, input_lon), (closest_hub['Latitude'], closest_hub['Longitude'])).kilometers, 2)
        hub_travel_time = get_travel_time((input_lat, input_lon), (closest_hub['Latitude'], closest_hub['Longitude']))
        _, church_index = church_tree.query([input_lat, input_lon])
        closest_church = church_df.iloc[church_index]
        church_distance_km = round(geodesic((input_lat, input_lon), (closest_church['Latitude'], closest_church['Longitude'])).kilometers, 2)
        church_travel_time = get_travel_time((input_lat, input_lon), (closest_church['Latitude'], closest_church['Longitude']))

        folium_map = generate_map(input_lat, input_lon, input_value, closest_hub, hub_distance_km, hub_travel_time, closest_church, church_distance_km, church_travel_time, voronoi_polygons, church_df, similarity_score)
        folium_map.save("static/voronoi_language_map.html")
        return render_template("index6.html", map_path="static/voronoi_language_map.html", country_image=country_image, barriers_info=barriers_info, similarity_score=similarity_score)

    return render_template('index6.html', map_path="static/voronoi_language_map.html")

if __name__ == "__main__":
    df, hub_coords, untranslated_coords, hub_locations = load_data()
    church_df = load_churches()
    land_mask = load_shapefile()
    church_tree = cKDTree(church_df[['Latitude', 'Longitude']].values)
    if os.path.exists("voronoi_cache.pkl"):
        with open("voronoi_cache.pkl", "rb") as f:
            voronoi_polygons = pickle.load(f)
            print("Loaded Voronoi polygons from cache.")
    else:
        voronoi_polygons = compute_voronoi(hub_locations, land_mask, hub_coords, df)
        with open("voronoi_cache.pkl", "wb") as f:
            pickle.dump(voronoi_polygons, f)
            print("Computed and cached Voronoi polygons.")
    
        # --- Voronoi Statistics ---
    num_regions = len(voronoi_polygons)

    # Track stats
    untranslated_counts = []
    only_untranslated_regions = 0
    redundant_translation_zones = 0
    only_untranslated_except_hub = 0
    regions_with_5plus_untranslated = 0


    for _, _, _, num_translated, num_untranslated, translated, untranslated in voronoi_polygons:
        if num_untranslated > 0:
            untranslated_counts.append(num_untranslated)
        if num_translated == 0 and num_untranslated > 0:
            only_untranslated_regions += 1
        if num_translated > 1:
            redundant_translation_zones += 1

    avg_untranslated_per_region = round(np.mean(untranslated_counts), 2) if untranslated_counts else 0
    
  
    print("\n--- Voronoi Region Statistics ---")
    red_regions = 0
    single_translated_regions = 0

    for idx, (hub_lang, poly, ratio, num_translated, num_untranslated, translated_langs, untranslated_langs) in enumerate(voronoi_polygons):
        if ratio < 0.2:
            red_regions += 1
        if num_translated == 1:
            single_translated_regions += 1
        if num_translated == 1 and translated == [hub_lang]:
            only_untranslated_except_hub += 1
        if num_untranslated >= 10:
            regions_with_5plus_untranslated += 1

    print("Starting Flask app...")
    app.run(debug=True)
