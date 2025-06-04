#!/usr/bin/env python
# coding: utf-8

# # I - Import des factures

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


get_ipython().system('apt update')

get_ipython().system('pip install --quiet pytesseract pdfplumber pdf2image pillow')


# ##I.1 - Extraction des factures

# In[ ]:


import pdfplumber
import os
import re

factures_path = '/content/drive/My Drive/Analyse des Ventes/Factures AUTENTIK'
liste_factures = [f for f in os.listdir(factures_path) if f.lower().endswith('.pdf')]

def nettoyer_texte(text):
    # Ajouter un espace après "du" dans les lignes BL N° :
    text = re.sub(r'(BL N° : \d+ du)(\d{2}/\d{2}/\d{4})', r'\1 \2', text)

    # Corriger les désignations coupées
    corrections = {
        r'\bcr pes\b': 'crêpes',
        r'\bcr pes Authentique\b': 'crêpes Authentique',
        r'\bg teaux\b': 'gâteaux',
        r'\bpaquet (s)': 'paquet(s)',
        # ajouter d'autres corrections si besoin
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)

    # Supprimer les lignes qui contiennent uniquement "g" ou des lignes vides ou inutiles
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip() == 'g' or line.strip() == '':
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def extraire_articles(facture_texte):
    lignes = facture_texte.split('\n')
    articles = []
    capture = False
    for ligne in lignes:
        ligne = ligne.strip()
        if "Référence Désignation Quantité P.U. HT Montant HT" in ligne:
            capture = True
            continue
        # Parasites
        if capture and (
            ligne.startswith("Total") or
            ligne.startswith("Taux") or
            ligne.startswith("Echéance") or
            ligne.startswith("BNP") or
            ligne == "" or
            ligne.startswith("SARL") or
            ligne.startswith("A reporter") or
            ligne.startswith("REPORT")
        ):
            break
        if capture:
            if ligne:
                articles.append(ligne)
    return articles

for i, nom_facture in enumerate(liste_factures, 1):
    chemin = os.path.join(factures_path, nom_facture)
    with pdfplumber.open(chemin) as pdf:
        texte_complet = ''
        for page in pdf.pages:
            texte_complet += page.extract_text() + '\n'
    texte_nettoye = nettoyer_texte(texte_complet)  # Nettoyage
    articles = extraire_articles(texte_nettoye)
    print(f"--- Articles facture #{i} : {nom_facture} ---")
    for art in articles:
        print(art)
    print("\n")




# In[ ]:


import pandas as pd

def parser_lignes_articles(articles):
    lignes_parsees = []
    date_bl = None

    for ligne in articles:
        # Détecter les lignes de type "BL N° : ..."
        match_bl = re.match(r'BL N° ?: \d+ du (\d{2}/\d{2}/\d{4})', ligne)
        if match_bl:
            date_bl = match_bl.group(1)
            continue

        # Séparer chaque ligne d'article par des espaces
        parts = ligne.split()
        if len(parts) < 5:
            continue  # ligne anormalement courte

        try:
            reference = parts[0]
            quantite = float(parts[-4].replace(',', '.'))
            prix_unitaire_ht = float(parts[-3].replace(',', '.'))
            # Les montants et codes divers sont dans les deux dernières colonnes
            designation = ' '.join(parts[1:-4])
        except ValueError:
            continue  # ignorer les lignes incorrectes

        lignes_parsees.append({
            'reference': reference,
            'designation': designation,
            'quantite': quantite,
            'prix_unitaire_ht': prix_unitaire_ht,
            'date': date_bl
        })

    return lignes_parsees

# Collecte de toutes les lignes d'articles de toutes les factures
toutes_lignes = []

for i, nom_facture in enumerate(liste_factures, 1):
    chemin = os.path.join(factures_path, nom_facture)
    with pdfplumber.open(chemin) as pdf:
        texte_complet = ''
        for page in pdf.pages:
            texte_complet += page.extract_text() + '\n'
    texte_nettoye = nettoyer_texte(texte_complet)
    articles = extraire_articles(texte_nettoye)
    lignes = parser_lignes_articles(articles)
    toutes_lignes.extend(lignes)

# Création de la DataFrame finale
df_factures = pd.DataFrame(toutes_lignes)

# Affichage pour vérification
print(df_factures.head())



# In[ ]:


pd.set_option('display.max_rows', None)
display(df_factures)
pd.reset_option('display.max_rows')


# #II - Traitement des données

# In[ ]:


import pandas as pd
from datetime import datetime, timedelta

# Préparation des données
df = df_factures.copy()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['semaine'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Séparation commandes / reprises
df['type_ligne'] = df['reference'].apply(lambda x: 'reprise' if x.startswith('R') else 'commande')
df['ref_base'] = df['reference'].apply(lambda x: x[1:] if x.startswith('R') else x)

# Fusion commandes et reprises
df_cmd = df[df['type_ligne'] == 'commande']
df_rep = df[df['type_ligne'] == 'reprise']

# Jointure pour obtenir les quantités vendues = commandées - reprises
df_merged = pd.merge(
    df_cmd,
    df_rep[['ref_base', 'semaine', 'quantite']],
    how='left',
    left_on=['reference', 'semaine'],
    right_on=['ref_base', 'semaine'],
    suffixes=('', '_reprise')
)
df_merged['quantite_reprise'] = df_merged['quantite_reprise'].fillna(0)
df_merged['quantite_vendue'] = df_merged['quantite'] - abs(df_merged['quantite_reprise'])

# Attribution de famille
df_merged['famille'] = 'Crêpes AUTENTIK'

# Df final
df_final = df_merged.groupby(['reference', 'designation']).agg(
    prix_unitaire_moyen=('prix_unitaire_ht', 'mean'),
    nb_commandes=('semaine', 'nunique'),
    quantites_commandees=('quantite', lambda x: list(x)),
    quantites_vendues=('quantite_vendue', lambda x: list(x)),
    dates=('semaine', lambda x: list(sorted(x.unique()))),
    famille=('famille', 'first')
).reset_index()

# Creation du champ ca_article (/sem.)
# len(ca_article)= nb semaines entre la première cmd et aujhourd'hui
toutes_les_semaines = pd.date_range(
    start=df['semaine'].min(),
    end=datetime.today(),
    freq='W-MON'
)

# Initialiser ca_article
def get_ca_par_semaine(reference, dates, quantites, prix):
    serie = [0.0 for _ in toutes_les_semaines]
    prix_unitaire = prix
    marge = 1.3
    for d, q in zip(dates, quantites):
        try:
            idx = toutes_les_semaines.get_loc(d)
            serie[idx] += q * prix_unitaire * marge
        except KeyError:
            continue
    return serie

df_final['ca_article'] = df_final.apply(
    lambda row: get_ca_par_semaine(
        row['reference'],
        row['dates'],
        row['quantites_vendues'],
        row['prix_unitaire_moyen']
    ),
    axis=1
)


# In[ ]:


#display(df_cmd)
#display(df_rep)
#display(df_merged)


# In[ ]:


display(df_final)


# In[ ]:


# Check up d'une ligne aléatoire
ligne_gx3 = df_final[df_final['reference'] == 'GX3'].iloc[0]

# Afficher les longueurs des listes
print("Longueur de quantites_commandees :", len(ligne_gx3['quantites_commandees']))
print("Longueur de quantites_vendues :", len(ligne_gx3['quantites_vendues']))
print("Longueur de dates :", len(ligne_gx3['dates']))
print("Longueur de ca_article :", len(ligne_gx3['ca_article']))


# #III - Visualisation graphique du CA

# In[ ]:


import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import timedelta

# Additionner semaine par semaine tous les ca_article
ca_semaine = np.sum(df_final['ca_article'].tolist(), axis=0)

# CA cumulé semaine par semaine
ca_cumule = np.cumsum(ca_semaine)


plus_vieille_date = min(df_final['dates'].apply(min))
plus_vieille_semaine = plus_vieille_date - pd.to_timedelta(plus_vieille_date.weekday(), unit='D')
semaines = [plus_vieille_semaine + timedelta(weeks=i) for i in range(len(ca_cumule))]

# Graphique de visualisation Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=semaines,
    y=ca_cumule,
    mode='lines+markers',
    name='CA Cumulé',
    line=dict(color='royalblue', width=3)
))

fig.update_layout(
    title="CA Cumulé - L'AUTENTIK",
    xaxis_title='Date',
    yaxis_title='Chiffre d’Affaires Cumulé (€)',
    hovermode='x unified',
    template='plotly_white'
)

fig.show()


# ### IV - Prédiction du CA 2025

# In[ ]:


from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

# Date de début : plus vieille facture
date_debut = plus_vieille_semaine

# Date de fin : dernier lundi de 2025
date_fin = datetime(2025, 12, 29)

# Plage temporelle d'étude
nb_semaines_totales = (date_fin - date_debut).days // 7 + 1

X = np.arange(len(ca_cumule)).reshape(-1, 1)
y = ca_cumule

# Initialisation du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

X_full = np.arange(nb_semaines_totales).reshape(-1, 1)
y_pred = model.predict(X_full)

# Générer la liste des semaines à rpz sur le graph
semaines_etude = [date_debut + timedelta(weeks=i) for i in range(nb_semaines_totales)]

# Graphique interactif plotly
fig = go.Figure()

# Courbe CA cumulé actuel
fig.add_trace(go.Scatter(
    x=semaines_etude[:len(y)],
    y=y,
    mode='lines+markers',
    name='CA Cumulé Observé',
    line=dict(color='royalblue')
))

# Courbe prévisionnelle
fig.add_trace(go.Scatter(
    x=semaines_etude,
    y=y_pred,
    mode='lines',
    name='Prédiction Linéaire (2025)',
    line=dict(color='firebrick')
))

fig.update_layout(
    title="Prédiction du CA Cumulé jusqu’à fin 2025 - L'AUTENTIK",
    xaxis_title='Semaine',
    yaxis_title='Chiffre d’Affaires Cumulé (€)',
    hovermode='x unified',
    template='plotly_white'
)

fig.show()



# ##IV.1 - Scoring de la régression

# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Prédictions du modèle sur les données d'entraînement
y_pred_eval = model.predict(X)

# Scorings
r2 = r2_score(y, y_pred_eval)
rmse = np.sqrt(mean_squared_error(y, y_pred_eval))
mae = mean_absolute_error(y, y_pred_eval)

print(f"R² : {r2:.4f}")
print(f"RMSE : {rmse:.2f} €")
print(f"MAE  : {mae:.2f} €")



# In[ ]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Colab_Notebooks/AnalyseVentes_SUPERGROUP.ipynb"')

