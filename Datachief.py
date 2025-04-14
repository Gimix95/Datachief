import pandas as pd
import unicodedata
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------------------
# Chargement des données depuis les fichiers Excel
# ---------------------------
event_file = "/home/gmartin/hhh/event_cleaned_10000.xlsx"
venue_file = "/home/gmartin/hhh/venue_+_address_joined.xlsx"

df_event = pd.read_excel(event_file)
df_venue = pd.read_excel(venue_file)

df = pd.merge(df_event, df_venue, left_on="venue_id", right_on="id")

print(df.columns)

df['local_start_date'] = pd.to_datetime(df['local_start_date'], errors='coerce')
df['local_end_date'] = pd.to_datetime(df['local_end_date'], errors='coerce')
df.dropna(subset=['local_start_date', 'local_end_date'], inplace=True)
df = df[df['local_end_date'] >= df['local_start_date']]

# Nettoyage des textes
def clean_text(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

df["name_clean"] = df["name_x"].apply(clean_text)
df["description_clean"] = df["description_x"].apply(clean_text)

# ---------------------------
# Filtrage temporel (±30 jours)
# ---------------------------
df['start_date_num'] = df['local_start_date'].astype('int64') // 10**9
df = df.sort_values(by='start_date_num')

# ---------------------------
# Optimisation spatiale avec KD-Tree
# ---------------------------
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df.dropna(subset=["latitude", "longitude"], inplace=True)

coords = list(zip(df["latitude"], df["longitude"]))
kdtree = KDTree(coords)

# ---------------------------
# Détection des doublons (textuel + géographique)
# ---------------------------
vectorizer = TfidfVectorizer().fit_transform(df["name_clean"])
cosine_sim = cosine_similarity(vectorizer)

vectorizer_desc = TfidfVectorizer().fit_transform(df["description_clean"].fillna(""))
cosine_sim_desc = cosine_similarity(vectorizer_desc)

SEUIL_TEXTE = 70
SEUIL_GEO = 0.5  # 500m
SEUIL_TEMPS = 30 * 86400  # 30 jours en secondes

doublons = []
for i in range(len(df)):
    lat1, lon1, date1 = df.iloc[i][["latitude", "longitude", "start_date_num"]]
    indices_proches = kdtree.query_ball_point((lat1, lon1), SEUIL_GEO / 111)  # Approximation km->deg
    
    for j in indices_proches:
        if i >= j:
            continue
        
        lat2, lon2, date2 = df.iloc[j][["latitude", "longitude", "start_date_num"]]
        
        # Vérification temporelle
        if abs(date1 - date2) > SEUIL_TEMPS:
            continue
        
        # Calcul des similarités
        name_sim = fuzz.token_sort_ratio(df.iloc[i]["name_clean"], df.iloc[j]["name_clean"])
        desc_sim = cosine_sim_desc[i, j] * 100
        dist = geodesic((lat1, lon1), (lat2, lon2)).km
        
        # Score final pondéré
        score = (0.5 * name_sim) + (0.3 * desc_sim) + (0.2 * (100 - dist * 100))
        
        if score >= 70:
            doublons.append({
                "Event 1": df.iloc[i]["name_x"],
                "Event 2": df.iloc[j]["name_x"],
                "Date 1": df.iloc[i]["local_start_date"],
                "Date 2": df.iloc[j]["local_start_date"],
                "Distance (km)": round(dist, 3),
                "Similarité Nom (%)": name_sim,
                "Similarité Description (%)": desc_sim,
                "Score (%)": round(score, 2)
            })

# ---------------------------
# Export des doublons détectés
# ---------------------------
df_doublons = pd.DataFrame(doublons)

# Application de styles
styled_df = df_doublons.style.highlight_max(color='lightgreen').highlight_min(color='lightcoral').format({'Distance (km)': '{:.2f}', 'Score (%)': '{:.2f}'})

# Affichage du DataFrame stylisé
print(styled_df)

# Ou sauvegarde dans un fichier HTML
styled_df.to_html("detected_duplicates_styled.html")

# Visualisation avec Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(df_doublons['Distance (km)'], df_doublons['Score (%)'])
plt.xlabel('Distance (km)')
plt.ylabel('Score (%)')
plt.title('Distance vs. Score des doublons détectés')
plt.grid(True)
plt.show()

# Tableaux interactifs avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_doublons.columns), align='left'),
    cells=dict(values=[df_doublons[col] for col in df_doublons.columns], align='left'))
])

fig.show()



