#!/usr/bin/env python
# coding: utf-8

# #I - Import des données

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# #I.1 - Lecture des factures

# In[ ]:


get_ipython().system('pip install pdfplumber')


# In[ ]:


import pdfplumber

chemin_pdf_test = '/content/drive/My Drive/Analyse des Ventes/Factures LBB/2025022014025353F_7636.PDF'


# Lecture des fichiers - test
with pdfplumber.open(chemin_pdf_test) as pdf:
    for page in pdf.pages:
        texte = page.extract_text()
        if texte:
            print("\n=== NOUVELLE PAGE ===\n")
            for ligne in texte.split('\n'):
                print(repr(ligne))


# In[ ]:


import pdfplumber
import re
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Fonctions d'extraction

def extraire_lignes_brutes_par_page(chemin_pdf):
    lignes_totales = []
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            lignes = page.extract_text().split('\n')
            lignes_totales.extend(lignes)
    return lignes_totales

def nettoyer_ligne(ligne):
    ligne = ligne.strip()
    ligne = re.sub(r'\s+', ' ', ligne)
    return ligne

pattern_ligne = re.compile(
    r'^(?P<code>\d{5,7})\s+'
    r'(?P<designation>.+?)\s+'
    r'(?P<gencod>\d{12,13})\s+'
    r'(?P<cond>\d+\s\w{2,4})\s+'
    r'(?P<quantite>\d+)\s+'
    r'(?P<unite>\w{2,4})\s+'
    r'(?P<prix_unitaire_ht>\d+[,.]\d+)\s+'
    r'(?P<montant_ht>\d+[,.]\d+)\s+'
    r'(?P<droits>[\d,.]+)'
)

def extraire_date_de_livraison(lignes):
    date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')
    for ligne in lignes:
        dates = date_pattern.findall(ligne)
        if len(dates) >= 2:
            return datetime.strptime(dates[1], '%d/%m/%Y').date()
    return None

def extraire_donnees_lignes(lignes):
    data = []
    for ligne in lignes:
        ligne = nettoyer_ligne(ligne)
        m = pattern_ligne.match(ligne)
        if m:
            data.append(m.groupdict())
    df = pd.DataFrame(data)
    if not df.empty:
        df['prix_unitaire_ht'] = df['prix_unitaire_ht'].str.replace(',', '.').astype(float)
        df['montant_ht'] = df['montant_ht'].str.replace(',', '.').astype(float)
        df['droits'] = df['droits'].str.replace(',', '.').astype(float)
        df['quantite'] = df['quantite'].astype(int)
        df['designation'] = df['designation'].str.strip()
        df['cond'] = df['cond'].str.strip()
        # Nettoyage (cid:176)
        df['designation'] = df['designation'].str.replace(r'\(cid:176\)', '°', regex=True)
        # Supprimer colonnes inutiles
        df = df.drop(columns=['cond', 'droits', 'unite', 'montant_ht'])
    return df

# === TRAITEMENT DE TOUS LES FICHIERS ===

# Chemin du dossier contenant les PDF (dans Google Drive monté)
repertoire_factures = '/content/drive/My Drive/Analyse des Ventes/Factures LBB'

# Lister tous les fichiers PDF
pdf_files = sorted([str(p) for p in Path(repertoire_factures).glob("*.PDF")])

print(f"{len(pdf_files)} fichiers trouvés.")

factures_concat = []

for chemin_pdf in pdf_files:
    print(f"Traitement : {chemin_pdf}")
    try:
        lignes = extraire_lignes_brutes_par_page(chemin_pdf)
        df_facture = extraire_donnees_lignes(lignes)
        date_livraison = extraire_date_de_livraison(lignes)
        df_facture['dates'] = date_livraison if date_livraison else pd.NaT
        df_facture['fichier'] = os.path.basename(chemin_pdf)
        factures_concat.append(df_facture)
    except Exception as e:
        print(f"❌ Erreur avec {chemin_pdf} : {e}")

# Fusionner tous les résultats
df_final = pd.concat(factures_concat, ignore_index=True)

# Affichage
print(f"{len(df_final)} lignes extraites au total.")
display(df_final)

df_final.to_csv("/content/factures_lbb_extraites.csv", index=False)


# In[ ]:


# S'assurer que 'date_livraison' est bien de type date
df_final['dates'] = pd.to_datetime(df_final['dates'], errors='coerce')

# Résumé
nb_refs_uniques = df_final['code'].nunique()
date_min = df_final['dates'].min()
date_max = df_final['dates'].max()

# Affichage
print(f"Nombre de références uniques : {nb_refs_uniques}")
print(f"Date de livraison la plus ancienne : {date_min.strftime('%d/%m/%Y') if pd.notnull(date_min) else 'Inconnue'}")
print(f"Date de livraison la plus récente  : {date_max.strftime('%d/%m/%Y') if pd.notnull(date_max) else 'Inconnue'}")



# In[ ]:


display(df_final)


# In[ ]:


display(df_final)


# In[ ]:


import pandas as pd
import numpy as np
from datetime import timedelta

# Normaliser les dates au lundi de chaque semaine
df_final['dates'] = pd.to_datetime(df_final['dates'])
df_final['dates_cmd'] = df_final['dates'].dt.to_period('W').apply(lambda r: r.start_time)
df_final['famille'] = 'Boissons LBB'

# Grouper les données par réf
df_grouped_lbb = df_final.groupby(['code', 'designation']).agg(
    prix_unitaire_moyen=('prix_unitaire_ht', 'mean'),
    quantites_commandees=('quantite', lambda x: list(x)),
    dates_commandes=('dates_cmd', lambda x: list(sorted(x))),
    famille=('famille', 'first')
).reset_index()

# Générer la plage temp. d'études

dates_valides = [min(d) for d in df_grouped_lbb['dates_commandes'] if isinstance(d, list) and d]
date_debut_globale = min(dates_valides)
toutes_les_semaines = pd.date_range(start=date_debut_globale, end=pd.to_datetime("today"), freq='W-MON')
nb_semaines = len(toutes_les_semaines)

# Fonction pour étaler les ventes - mm que celle de 'AnalyserVentes_SUPERGROUP'
def etaler_ventes(ref, date_debut_globale, nb_semaines):
    commandes = [pd.to_datetime(d) for d in ref['dates_commandes']]
    quantites = ref['quantites_commandees']
    ventes = [0] * nb_semaines

    if len(commandes) < 2:
        date_debut = commandes[0]
        index_debut = (date_debut - date_debut_globale).days // 7
        for j in range(17):
            if 0 <= index_debut + j < nb_semaines:
                ventes[index_debut + j] += quantites[0] / 17
    else:
        for i in range(len(commandes)):
            date_debut = commandes[i]
            if i < len(commandes) - 1:
                date_fin = commandes[i + 1]
            else:
                deltas = [(commandes[j+1] - commandes[j]).days // 7 for j in range(len(commandes)-1)]
                freq_moy = int(np.mean(deltas)) if deltas else 17
                date_fin = date_debut + timedelta(weeks=freq_moy)

            semaines = pd.date_range(start=date_debut, end=date_fin, freq='W-MON')
            qte_par_semaine = quantites[i] / len(semaines) if len(semaines) > 0 else 0
            for semaine in semaines:
                idx = (semaine - date_debut_globale).days // 7
                if 0 <= idx < nb_semaines:
                    ventes[idx] += qte_par_semaine
    return ventes

# Appliquer l’étalement
df_grouped_lbb['quantites_vendues'] = df_grouped_lbb.apply(
    lambda row: etaler_ventes(row, date_debut_globale, nb_semaines),
    axis=1
)

# Calcul du CA par semaine avec marge
def get_ca_article(ventes, prix_unitaire, marge=1.3):
    return [q * prix_unitaire * marge for q in ventes]

df_grouped_lbb['ca_article'] = df_grouped_lbb.apply(
    lambda row: get_ca_article(row['quantites_vendues'], row['prix_unitaire_moyen']),
    axis=1
)



# In[ ]:


df_grouped_lbb


# In[ ]:


import plotly.graph_objs as go
from datetime import timedelta

# Additionner semaine par semaine tous les ca_article
ca_semaine = np.sum(df_grouped_lbb['ca_article'].tolist(), axis=0)

# CA cumulé semaine par semaine
ca_cumule = np.cumsum(ca_semaine)

# Générer la liste des dates hebdomadaires à partir de la date_debut_globale
semaines = [date_debut_globale + timedelta(weeks=i) for i in range(len(ca_cumule))]

# Création du graphique Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=semaines,
    y=ca_cumule,
    mode='lines+markers',
    name='CA Cumulé',
    line=dict(color='royalblue', width=3)
))

fig.update_layout(
    title="CA Cumulé - LBB",
    xaxis_title='Date',
    yaxis_title='Chiffre d’Affaires Cumulé (€)',
    hovermode='x unified',
    template='plotly_white'
)

fig.show()


# In[ ]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# préparation des données
X = np.array([(d - date_debut_globale).days for d in semaines]).reshape(-1, 1)
y = ca_cumule

# entraînement
model = LinearRegression()
model.fit(X, y)

# prédiction jusqu’à fin 2025
semaines_2025 = pd.date_range(start=semaines[0], end='2025-12-29', freq='W-MON')
X_pred = np.array([(d - date_debut_globale).days for d in semaines_2025]).reshape(-1, 1)
y_pred = model.predict(X_pred)

# visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=semaines, y=ca_cumule, mode='lines+markers', name='CA Observé', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=semaines_2025, y=y_pred, mode='lines', name='CA Prédit', line=dict(color='orange', dash='dash')))
fig.update_layout(
    title="CA Cumulé - Régression linéaire (prévision jusqu’à fin 2025)",
    xaxis_title='Date',
    yaxis_title='CA Cumulé (€)',
    template='plotly_white'
)
fig.show()



# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


date_debut_globale = pd.Timestamp('2024-11-04')

X = np.array([(d - date_debut_globale).days for d in semaines]).reshape(-1, 1)
y = np.array(ca_cumule)

# Entraîner la régression linéaire
model = LinearRegression()
model.fit(X, y)

# Étendre la période de prédiction jusqu'au 31 décembre 2025 (semaines complètes, lundi)
semaines_etendues = pd.date_range(start=semaines[0], end='2025-12-28', freq='W-MON')
X_pred = np.array([(d - date_debut_globale).days for d in semaines_etendues]).reshape(-1, 1)
y_pred = model.predict(X_pred)

# Date initial
date_ref = pd.Timestamp('2025-01-01')
idx_debut_2025 = semaines_etendues.get_indexer([date_ref], method='bfill')[0]

# Offset à appliquer pour initaliser le ca à 0 au 01/01/2025
offset_reel = y[idx_debut_2025] if idx_debut_2025 < len(y) else 0
offset_pred = y_pred[idx_debut_2025]

# Recalage des courbes : soustraire offset pour démarrer à 0 au 01/01/2025
y_recale = y - offset_reel
y_pred_recale = y_pred - offset_pred

# Plot avec zoom sur la période 2025
fig = go.Figure()

# Courbe CA réel recalée
fig.add_trace(go.Scatter(
    x=semaines,
    y=y_recale,
    mode='lines+markers',
    name='CA cumulé réel recalé',
    line=dict(color='blue')
))

# Courbe CA prédite recalée
fig.add_trace(go.Scatter(
    x=semaines_etendues,
    y=y_pred_recale,
    mode='lines',
    name='CA cumulé prédit recalé',
    line=dict(color='orange', dash='dash')
))

# Zoom affichage à partir du 01/01/2025
fig.update_layout(
    title="CA Cumulé - Régression linéaire recalée (affichage 2025)",
    xaxis_title='Date',
    yaxis_title='CA cumulé (€) (décalé à 0 au 01/01/2025)',
    template='plotly_white',
    xaxis=dict(range=[date_ref, semaines_etendues[-1]])
)

fig.show()

