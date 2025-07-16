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
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)
geolocator = Nominatim(user_agent="lang_mapper")

from ollama import chat, ChatResponse

def fetch_barriers_to_entry(lang_name, country):
    #commenting out temporaririly 
    # try:
    #     response: ChatResponse = chat(model='llama3', messages=[
    #         {
    #             'role': 'user',
    #             'content': f'What are the barriers to entry for the {lang_name} language in {country}? Limit response to 3 short bullet points. Base the response on an accredited website, like Harvard Business Review. Also, if available, please provide the Government restriction score as well. Please ensure you do not say anything like "This language does not exist". If you are unable to find information for the language give general details related to the country or closest city. Please share only one value for government restriction score. Also make a table that shows results for translation costs in this area.'
    #         },
    #     ])
    #     return response['message']['content']
    # except Exception as e:
    #     print(f"Error fetching LLM barriers: {e}")
    #     return None
    return "ABCD LLM Response here."

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

#def generate_cost_breakdown_chart(base_cost, complexity, accessibility, capacity): #old version matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    # Calculations
    complexity_val = base_cost * (complexity - 1)
    accessibility_val = base_cost * complexity * (accessibility - 1)
    capacity_val = base_cost * complexity * accessibility * (capacity - 1)
    total_cost = round(base_cost * complexity * accessibility * capacity)

    labels = ["Base Cost", "Complexity", "Accessibility", "Capacity Adj."]
    values = [base_cost, complexity_val, accessibility_val, capacity_val]
    colors = ['#6c757d', '#f39c12', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor='black')

    # Format currency on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.03,
                f"${int(height):,}", ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_ylabel("USD", fontsize=11)
    ax.set_title("Translation Cost Breakdown", fontsize=14, weight='bold')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    plt.xticks(fontsize=10)
    plt.tight_layout()

    # Add white space below chart
    fig.subplots_adjust(bottom=0.2)

    # Save chart to image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_chart = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)

    # Return both chart and clean total string
    total_cost_str = f"${total_cost:,.0f}"
    return encoded_chart, total_cost_str

import plotly.graph_objects as go
import plotly.io as pio

def generate_cost_breakdown_chart(base_cost, complexity, accessibility, capacity):
    # Compute components
    complexity_val = base_cost * (complexity - 1)
    accessibility_val = base_cost * complexity * (accessibility - 1)
    capacity_val = base_cost * complexity * accessibility * (capacity - 1)
    total_cost = round(base_cost * complexity * accessibility * capacity)

    labels = ["Base Cost", "Complexity", "Accessibility", "Capacity Adj."]
    values = [base_cost, complexity_val, accessibility_val, capacity_val]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition='auto',
            hovertemplate='%{x}<br>Cost: %{text}<extra></extra>',
        )
    ])

    fig.update_layout(
        title="Translation Cost Breakdown",
        yaxis_title="Cost (USD)",
        xaxis_title="Component",
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9",
        font=dict(size=14),
        margin=dict(l=40, r=20, t=50, b=50),
        height=400
    )

    # Return full HTML div
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    total_cost_str = f"${total_cost:,.0f}"
    return chart_html, total_cost_str


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


def generate_map(input_lat, input_lon, input_language_code, closest_hub, hub_distance_km, hub_travel_time, closest_church, church_distance_km, church_travel_time, voronoi_polygons, church_df, similarity_score,estimated_cost):
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
               f"<b>Closest Church:</b> {closest_church['Name']} ({church_distance_km} km, {church_travel_time} min)<br>"
               f"<b>Estimated Translation Cost:</b> ${estimated_cost}"),
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

    vec1_all, vec2_all = [], []

    for feature in features:
        try:
            data = l2v.get_features([lang1, lang2], feature)
        except:
            return None  # silently skip and return nothing

        vec1 = data.get(lang1)
        vec2 = data.get(lang2)
        if vec1 is None or vec2 is None:
            continue

        filtered = [(float(x), float(y)) for x, y in zip(vec1, vec2) if x != '--' and y != '--']
        if filtered:
            v1_filtered, v2_filtered = zip(*filtered)
            vec1_all.extend(v1_filtered)
            vec2_all.extend(v2_filtered)

    if not vec1_all or not vec2_all:
        return None

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    return round(cosine_similarity(np.array(vec1_all).reshape(1, -1),
                                   np.array(vec2_all).reshape(1, -1))[0][0], 2)



# def get_language_similarity(lang1, lang2, features=None):
#     if features is None:
#         features = ["syntax_knn", "phonology_average", "inventory_average", "fam"]
#     valid_features = l2v.available_feature_sets()
#     for f in features:
#         if f not in valid_features:
#             raise ValueError(f"Feature set '{f}' is not available.")
#     vec1_all, vec2_all = [], []
#     for feature in features:
#         data = l2v.get_features([lang1, lang2], feature)
#         vec1, vec2 = data.get(lang1), data.get(lang2)
#         if vec1 is None or vec2 is None:
#             continue
#         filtered = [(float(x), float(y)) for x, y in zip(vec1, vec2) if x != '--' and y != '--']
#         if filtered:
#             v1_filtered, v2_filtered = zip(*filtered)
#             vec1_all.extend(v1_filtered)
#             vec2_all.extend(v2_filtered)
#     if not vec1_all or not vec2_all:
#         raise RuntimeError("No usable features found for comparison.")
#     return round(cosine_similarity(np.array(vec1_all).reshape(1, -1), np.array(vec2_all).reshape(1, -1))[0][0], 2)

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
                return render_template('index7.html', error=f"Language code '{input_value}' not found.")
            input_lat, input_lon = input_language.iloc[0][['Latitude', 'Longitude']]
            lang_row = input_language.iloc[0]
            print("Fetching barriers to entry from LLM...")
            barriers_info = fetch_barriers_to_entry(lang_row['LanguageName'], lang_row['Country'])

        elif input_type == 'city':
            coords = get_coordinates_from_city(input_value)
            if not coords:
                return render_template('index7.html', error=f"City '{input_value}' not found.")
            input_lat, input_lon = coords
            input_value = get_nearest_language_code(input_lat, input_lon)
            lang_row = df[df['Language Code'] == input_value].iloc[0]
            print("Fetching barriers to entry from LLM...")
            barriers_info = fetch_barriers_to_entry(lang_row['LanguageName'], lang_row['Country'])

        elif input_type == 'country':
            if input_value not in df['Country'].dropna().unique():
                return render_template('index7.html', error=f"Country '{input_value}' not found.")
            country_image = generate_country_pie_chart(input_value)
            return render_template("index7.html", country_image=country_image, map_path="static/voronoi_language_map.html")

        else:
            return render_template('index7.html', error="Invalid input type.")

        _, hub_index = cKDTree(hub_locations).query([input_lat, input_lon])
        print("Finding nearest hub language...")
        closest_hub = hub_coords.iloc[hub_index]
        print("Finding similarity score...")
        try:
            similarity_score = get_language_similarity(input_value, closest_hub['Language Code'])
        except ValueError as ve:
            print(str(ve))
            similarity_score = 0.5  # fallback default will need to try to come up with a btter idea here

        print("Finding travel distance and travel time...")
        hub_distance_km = round(geodesic((input_lat, input_lon), (closest_hub['Latitude'], closest_hub['Longitude'])).kilometers, 2)
        hub_travel_time = get_travel_time((input_lat, input_lon), (closest_hub['Latitude'], closest_hub['Longitude']))
        _, church_index = church_tree.query([input_lat, input_lon])
        closest_church = church_df.iloc[church_index]
        church_distance_km = round(geodesic((input_lat, input_lon), (closest_church['Latitude'], closest_church['Longitude'])).kilometers, 2)
        church_travel_time = get_travel_time((input_lat, input_lon), (closest_church['Latitude'], closest_church['Longitude']))

        # Count churches within 50 km
        church_coords = church_df[['Latitude', 'Longitude']].values
        distances = [geodesic((input_lat, input_lon), (lat, lon)).kilometers for lat, lon in church_coords]
        num_churches_nearby = sum(d <= 50 for d in distances)

        estimated_cost, base_cost, complexity, accessibility, capacity = estimate_translation_cost(
            similarity_score,
            hub_travel_time,
            num_churches_nearby,
            barriers_info
        )

        cost_chart_html, total_cost_str = generate_cost_breakdown_chart(base_cost, complexity, accessibility, capacity)


        folium_map = generate_map(input_lat, input_lon, input_value, closest_hub, hub_distance_km, hub_travel_time, closest_church, church_distance_km, church_travel_time, voronoi_polygons, church_df, similarity_score,estimated_cost)
        folium_map.save("static/voronoi_language_map.html")
        
        response = requests.get("https://api.flightapi.io/onewaytrip/6855816c8a2aa0fa24ef7369/DFW/DRW/2025-06-20/1/0/0/Economy/USD")
        print(response.json())
        return render_template(
            "index7.html",
            map_path="static/voronoi_language_map.html",
            country_image=country_image,
            barriers_info=barriers_info,
            similarity_score=similarity_score,
            estimated_cost=estimated_cost,
            cost_chart_html=cost_chart_html,
            total_cost_str=total_cost_str
)

    return render_template('index7.html', map_path="static/voronoi_language_map.html")

#def rank_voronoi_regions_in_australia(voronoi_polygons, df, church_df):
    """
    Ranks Voronoi regions located in Australia by impact (untranslated count / estimated cost).
    """
    from geopy.distance import geodesic

    results = []

    for region in voronoi_polygons:
        hub_lang, poly, ratio, num_translated, num_untranslated, translated, untranslated = region

        # Get metadata for the hub
        hub_row = df[df['Language Code'] == hub_lang]
        if hub_row.empty or hub_row.iloc[0]['Country'] != 'Australia':
            continue

        hub_lat, hub_lon = hub_row.iloc[0][['Latitude', 'Longitude']]

        # Get average location of untranslated languages in this region
        coords_df = df[df['Language Code'].isin(untranslated)][['Latitude', 'Longitude']]
        if coords_df.empty:
            continue
        avg_lat, avg_lon = coords_df.mean()

        # Calculate travel time to hub
        travel_time = get_travel_time((avg_lat, avg_lon), (hub_lat, hub_lon)) or 120

        # Count churches within 50 km
        church_coords = church_df[['Latitude', 'Longitude']].values
        distances = [geodesic((avg_lat, avg_lon), (lat, lon)).km for lat, lon in church_coords]
        num_churches_nearby = sum(d <= 50 for d in distances)

        # Estimate similarity between hub and first untranslated language (proxy)
        try:
            sim_score = get_language_similarity(untranslated[0], hub_lang) if untranslated else 0.5
        except ValueError as ve:
            print(str(ve))
            sim_score = 0.5  # fallback default - need to find something btter
        if sim_score is None:
            sim_score = 0.5  # or skip, or set default

        # Estimate translation cost
        cost, _, _, _, _ = estimate_translation_cost(sim_score, travel_time, num_churches_nearby, "")

        # Compute impact score
        impact_score = num_untranslated / cost if cost > 0 else 0
        print("Hub Language:", hub_lang,
        "Untranslated Count:", num_untranslated,
        "Estimated Cost:", cost,
        "Similarity Score:", round(sim_score, 2),
        "Travel Time (min):", round(travel_time, 1),
        "Churches Nearby:", num_churches_nearby,
        "Impact Score:", round(impact_score, 6))

        results.append({
            "Hub Language": hub_lang,
            "Untranslated Count": num_untranslated,
            "Estimated Cost": cost,
            "Similarity Score": round(sim_score, 2),
            "Travel Time (min)": round(travel_time, 1),
            "Churches Nearby": num_churches_nearby,
            "Impact Score": round(impact_score, 6)
        })

    ranked_df = pd.DataFrame(results).sort_values(by="Impact Score", ascending=False).reset_index(drop=True)
    return ranked_df

def rank_voronoi_regions_in_australia(voronoi_polygons, df, church_df):
    """
    Ranks Voronoi regions located in Australia by impact (untranslated count / estimated cost).
    Uses average similarity and travel time across all untranslated languages in the region.
    """
    from geopy.distance import geodesic

    results = []

    for region in voronoi_polygons:
        hub_lang, poly, ratio, num_translated, num_untranslated, translated, untranslated = region

        # Get metadata for the hub
        hub_row = df[df['Language Code'] == hub_lang]
        if hub_row.empty or hub_row.iloc[0]['Country'] != 'Australia':
            continue

        hub_lat, hub_lon = hub_row.iloc[0][['Latitude', 'Longitude']]

        # Coordinates of untranslated languages
        untranslated_df = df[df['Language Code'].isin(untranslated)].copy()
        if untranslated_df.empty:
            continue

        # --- Average Similarity Score ---
        similarity_scores = []
        for lang_code in untranslated:
            try:
                score = get_language_similarity(lang_code, hub_lang)
                print(score)
                if score is not None:
                    similarity_scores.append(score)
            except:
                continue

        if similarity_scores:
            avg_similarity = round(sum(similarity_scores) / len(similarity_scores), 2)
        else:
            avg_similarity = 0.5  # fallback

        # --- Average Travel Time ---
        travel_times = []
        for _, row in untranslated_df.iterrows():
            try:
                travel_time = get_travel_time(
                    (row['Latitude'], row['Longitude']), 
                    (hub_lat, hub_lon)
                )
                if travel_time:
                    travel_times.append(travel_time)
                    print(travel_time)
            except:
                continue

        if travel_times:
            avg_travel_time = round(sum(travel_times) / len(travel_times), 1)
        else:
            avg_travel_time = 120  # fallback

        from shapely.geometry import Point

        # Filter churches within the Voronoi polygon
        church_points = [Point(lon, lat) for lat, lon in church_df[['Latitude', 'Longitude']].values]
        churches_in_region = [pt for pt in church_points if poly.contains(pt)]
        num_churches_nearby = len(churches_in_region)

        # --- Estimate translation cost ---
        cost, _, _, _, _ = estimate_translation_cost(avg_similarity, avg_travel_time, num_churches_nearby, "")

        # --- Compute impact ---
        impact_score = num_untranslated / cost if cost > 0 else 0

        results.append({
            "Hub Language": hub_lang,
            "Untranslated Count": num_untranslated,
            "Estimated Cost": cost,
            "Similarity Score": avg_similarity,
            "Travel Time (min)": avg_travel_time,
            "Churches Nearby": num_churches_nearby,
            "Impact Score": round(impact_score, 6)
        })

    ranked_df = pd.DataFrame(results).sort_values(by="Impact Score", ascending=False).reset_index(drop=True)
    return ranked_df

def count_voronoi_regions_in_australia(voronoi_polygons, df):
    """
    Returns the number of Voronoi regions where the hub language is located in Australia.
    """
    count = 0
    for region in voronoi_polygons:
        hub_lang = region[0]
        hub_row = df[df['Language Code'] == hub_lang]
        if not hub_row.empty and hub_row.iloc[0]['Country'] == 'Australia':
            count += 1
    return count


def estimate_translation_cost(similarity_score, travel_time_minutes, num_churches, barriers_text, base_cost=300000):
    complexity = 1 + (1 - similarity_score)
    accessibility = 1 + (travel_time_minutes / 120 if travel_time_minutes else 1)
    
    if num_churches >= 5:
        capacity = 0.85
    elif "government restrictions" in (barriers_text or "").lower(): #change the text to a score maybe?
        capacity = 1.2
    else:
        capacity = 1.0

    total_cost = round(base_cost * complexity * accessibility * capacity, -3)
    return total_cost, base_cost, complexity, accessibility, capacity


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

    australia_region_count = count_voronoi_regions_in_australia(voronoi_polygons, df)
    print(f"Number of Voronoi regions in Australia: {australia_region_count}")


    # --- Rank Australian Regions by Impact ---
    ranked_aus_df = rank_voronoi_regions_in_australia(voronoi_polygons, df, church_df)
    print("\nTop Australian Regions by Impact:\n", ranked_aus_df.head())
    ranked_aus_df.to_csv("ranked_australian_regions.csv", index=False)

    print("Starting Flask app...")
    app.run(debug=True)
