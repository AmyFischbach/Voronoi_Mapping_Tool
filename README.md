# Geo-Mapping Tool with Voronoi Diagraming

## Project Overview

We utlized Python and HTML to build a site that creates a Voronoi diagram to highlight regions of interest for translation projects. The Voronoi color codes regions with high percentages of languages with untranslated Bibles, along with listing other factors for users to consider when deciding how to allocate resources to a project if it includes this region. An LLM feature is also included to show Barriers to Entry for the regaion of the hub language selected. 

---

## What's Included in this Project

1. Python Code the builds the tool based on language selection (includes HTML and LLM feature)
2. User Guide to work the code and download any necessary libraries.


---------
# 🌍 Language Mapping Tool for Bible Translation Strategy
This Flask-based geospatial analytics tool helps organizations make data-driven decisions for prioritizing Bible translation efforts. It visualizes the geographic distribution of hub and untranslated languages using Voronoi diagrams, evaluates proximity to churches and other locations, computes language similarity scores, and integrates LLM-based insights to uncover barriers to entry for translation work.

## 🚀 Features
- Voronoi Region Mapping: Automatically generates Voronoi polygons around hub languages based on their geographic coordinates.

- Interactive Folium Map: Visualizes translated/untranslated languages, churches, and proximity zones on a map with clustered markers and custom tooltips.

- Language Accessibility Insights:

  - Computes travel time and distance to nearest translation hub and church.

  - Displays a similarity score between the selected language and its nearest hub using lang2vec.

  - Fetches LLM-based barriers to entry using Ollama's llama3 model.

- Country Overview: Generates pie charts showing the translation status for all languages in a selected country.

- Input Flexibility: Accepts three input types—language code, city name, or country—for dynamic exploration.

## 🛠️ Dependencies
Install all required libraries using:

```pip install <libraryname>```

Key Libraries Used:

- Flask
- Folium + MarkerCluster
- Pandas, NumPy
- Shapely, Geopy, Scikit-learn
- lang2vec
- Ollama for LLM queries
- Scipy for Voronoi & KDTree

## 💡 Usage
#### 1. Place the necessary input files in the project directory:

- ProgressBible Data with Coordinates.xlsx
- churches_by_country.csv
- ne_10m_land.shp (plus associated .shx, .dbf files)

#### 2. Run the application:
   
```python MappingTool_SIL.py```

#### 3. Open http://127.0.0.1:5000 in your browser.

#### 4. Enter the language code/city that you wish to see insights for in the search box 

## 🧠 Insights Provided
- Voronoi Region Stats: Quantitative breakdown of translation coverage across regions.
- Language Similarity: Identifies how close an untranslated language is to an existing hub.
- Barriers to Entry: Leverages LLM output to list top 3 barriers to entry per input.

 

