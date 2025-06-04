#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


#packages à installer
get_ipython().system('pip install pdfplumber')


# In[ ]:


import os
import pdfplumber

factures_supergroup = '/content/drive/My Drive/Analyse des Ventes/Factures APDV'

# Liste tous les fichiers PDF dans le dossier
fichiers_pdf = [
    f for f in os.listdir(factures_supergroup)
    if f.lower().endswith('.pdf')
]

print("📄 Factures trouvées :", fichiers_pdf)

# Parcours et affiche le contenu de chaque facture
for fichier in fichiers_pdf:
    chemin_pdf = os.path.join(factures_supergroup, fichier)
    print(f"\n\n===== Contenu de la facture : {fichier} =====\n")

    with pdfplumber.open(chemin_pdf) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"\n----- PAGE {i+1} -----\n")
            texte = page.extract_text()
            if texte:
                print(texte)
            else:
                print("[Page sans texte détecté]")


# In[ ]:


import os
import pdfplumber
import pandas as pd
import re

# Dossier contenant les factures
factures_supergroup = '/content/drive/My Drive/Analyse des Ventes/Factures APDV'

# Liste tous les fichiers PDF dans le dossier
fichiers_pdf = [
    f for f in os.listdir(factures_supergroup)
    if f.lower().endswith('.pdf')
]

print("📄 Factures trouvées :", fichiers_pdf)

# Liste pour stocker tous les produits extraits
toutes_les_lignes = []

# Parcours de chaque facture
for fichier in fichiers_pdf:
    chemin_pdf = os.path.join(factures_supergroup, fichier)
    print(f"\n\n===== Traitement de la facture : {fichier} =====\n")

    texte_complet = ""
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            texte = page.extract_text()
            if texte:
                texte_complet += texte + "\n"
            else:
                print(f"[Page sans texte détectée dans {fichier}]")

    # Nettoyage du texte
    texte_clean = re.sub(r'-{5,} PAGE \d+ -{5,}', '', texte_complet)
    lignes = texte_clean.splitlines()

    # Mémorise les positions des commandes dans le texte
    dates_positions = []
    for idx, ligne in enumerate(lignes):
        match_date = re.search(r'Commande n° \d+ du (\d{2}/\d{2}/\d{4})', ligne)
        if match_date:
            date = match_date.group(1)
            dates_positions.append((idx, date))

    # Extraction des lignes produits
    for i, ligne in enumerate(lignes):
        match = re.match(r'^(.*)Bouteille\s+0\.75\s+L\s+(\d+,\d{2})\s+€\s+(\d+)\s+([\d,]+)', ligne)
        if match:
            designation = match.group(1).strip()
            prix_unitaire = float(match.group(2).replace(',', '.'))
            quantite = int(match.group(3))
            total_ht = float(match.group(4).replace(',', '.'))

            # Gencode dans les lignes suivantes
            gencode = None
            for j in range(i+1, i+5):
                if j < len(lignes):
                    gencode_match = re.search(r'Gencode.*?:\s*(\d+)', lignes[j])
                    if gencode_match:
                        gencode = gencode_match.group(1)
                        break

            # Trouver la dernière date "Commande n°... du ..." avant cette ligne
            date_commande = None
            for pos, date in reversed(dates_positions):
                if pos < i:
                    date_commande = date
                    break

            # Si aucune date avant, on prend la première disponible
            if date_commande is None and dates_positions:
                date_commande = dates_positions[0][1]

            sous_famille = designation.split()[0] if designation else None

            toutes_les_lignes.append({
                'fichier': fichier,
                'date_commande': date_commande,
                'designation': designation,
                'volume': '0.75 L',
                'prix_unitaire_ht': prix_unitaire,
                'quantite': quantite,
                'total_ht': total_ht,
                'gencode': gencode,
                'famille': 'Boissons',
                'sous_famille': sous_famille
            })

# Création du DataFrame global
df_factures_apdv = pd.DataFrame(toutes_les_lignes)

# Export CSV
df_factures_apdv.to_csv('/content/factures_apdv_extraites.csv', index=False)

print("\n✅ Extraction terminée. Fichier CSV sauvegardé sous /content/factures_apdv_extraites.csv")
display(df_factures_apdv)


# In[ ]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df_factures_apdv)


# In[ ]:


import pandas as pd

# Copie de travail
df = df_factures_apdv.copy()

# Conversion des dates en datetime
df['date_commande'] = pd.to_datetime(df['date_commande'], dayfirst=True)

df = df.rename(columns={'gencode': 'code'})

# Groupement
df_grouped_apdv = df.groupby(['code', 'designation'], as_index=False).agg({
    'prix_unitaire_ht': 'mean',
    'quantite': list,
    'date_commande': list,
    'famille': 'first',
    'sous_famille': 'first'
})

# Renommage et création des colonnes
df_grouped_apdv = df_grouped_apdv.rename(columns={
    'prix_unitaire_ht': 'prix_unitaire_moyen',
    'quantite': 'quantites_commandees',
    'date_commande': 'dates_commandes'
})

# Nombre de commandes = longueur de la liste des quantités
df_grouped_apdv['nb_commandes'] = df_grouped_apdv['quantites_commandees'].apply(len)

# Réorganisation des colonnes (ordre)
df_grouped_apdv = df_grouped_apdv[[
    'code', 'designation', 'prix_unitaire_moyen', 'nb_commandes',
    'quantites_commandees', 'dates_commandes', 'famille', 'sous_famille'
]]


# In[ ]:


display(df_grouped_apdv)


# In[ ]:


from datetime import timedelta
import numpy as np

# Définir la date de début globale et le nombre de semaines
date_debut_globale = min(df_grouped_apdv['dates_commandes'].apply(min))
date_fin_globale = pd.to_datetime("2025-12-31")
nb_semaines = (date_fin_globale - date_debut_globale).days // 7 + 1

# Fonction pour lisser les ventes - mm que celle dans 'AnalyseVentes_SUPERGROUP'
def etaler_ventes(ref, date_debut_globale, nb_semaines):
    commandes = [pd.to_datetime(d) for d in ref['dates_commandes']]
    quantites = ref['quantites_commandees']
    ventes = [0] * nb_semaines

    if len(commandes) < 2:
        date_debut = commandes[0]
        index_debut = (date_debut - date_debut_globale).days // 7
        for j in range(6):
            if 0 <= index_debut + j < nb_semaines:
                ventes[index_debut + j] += quantites[0] / 6
    else:
        for i in range(len(commandes)):
            date_debut = commandes[i]
            if i < len(commandes) - 1:
                date_fin = commandes[i + 1]
            else:
                deltas = [(commandes[j+1] - commandes[j]).days // 7 for j in range(len(commandes)-1)]
                freq_moy = int(np.mean(deltas)) if deltas else 6
                date_fin = date_debut + timedelta(weeks=freq_moy)

            semaines = pd.date_range(start=date_debut, end=date_fin, freq='W-MON')
            qte_par_semaine = quantites[i] / len(semaines) if len(semaines) > 0 else 0
            for semaine in semaines:
                idx = (semaine - date_debut_globale).days // 7
                if 0 <= idx < nb_semaines:
                    ventes[idx] += qte_par_semaine
    return ventes

# Appliquer à chaque ligne pour créer les colonnes 'ventes_etalées' et 'ca_article'
df_grouped_apdv['quantites_vendues'] = df_grouped_apdv.apply(
    lambda row: etaler_ventes(row, date_debut_globale, nb_semaines), axis=1
)

df_grouped_apdv['ca_article'] = df_grouped_apdv.apply(
    lambda row: [q * row['prix_unitaire_moyen'] * 1.3 for q in row['quantites_vendues']], axis=1
)


# In[ ]:


display(df_grouped_apdv)


# In[ ]:


# Préd inutile et inintéressante

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

# Copie de sécurité
df_apdv = df_grouped_apdv.copy()

date_debut_globale = datetime(2024, 1, 1)
nb_semaines = len(df_apdv['ca_article'].iloc[0])  # on suppose que toutes les lignes ont même longueur

# CA hebdomadaire
ca_total_hebdo = np.sum(df_apdv['ca_article'].tolist(), axis=0)

# Dates associées à chaque semaine
dates_semaine = [date_debut_globale + timedelta(weeks=i) for i in range(nb_semaines)]

X = np.arange(len(ca_total_hebdo)).reshape(-1, 1)
y = ca_total_hebdo

# Entraînement Modèle linéaire
modele = LinearRegression()
modele.fit(X, y)

y_pred = modele.predict(X)

# Prédiction pour les 4 semaines futures
semaines_futures = np.arange(len(ca_total_hebdo), len(ca_total_hebdo) + 4).reshape(-1, 1)
predictions_futures = modele.predict(semaines_futures)
dates_futures = [dates_semaine[-1] + timedelta(weeks=i+1) for i in range(4)]

# Affichage avec matplotlib
plt.figure(figsize=(10, 5))
plt.plot(dates_semaine, y, label='CA réel', marker='o')
plt.plot(dates_semaine, y_pred, label='Prédiction (historique)', linestyle='--')
plt.plot(dates_futures, predictions_futures, label='Prédiction (futur)', marker='x')
plt.title("Prédiction du chiffre d'affaires hebdomadaire (APDV)")
plt.xlabel("Semaine")
plt.ylabel("Chiffre d'affaires HT (€)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Métriques
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"MAE = {mae:.2f} €")
print(f"RMSE = {rmse:.2f} €")


# In[ ]:


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


df = df_grouped_apdv.copy()


all_dates = []
for dates_list in df['dates_commandes']:
    all_dates.extend(pd.to_datetime(dates_list))

date_debut = min(all_dates).normalize()

# Calculer le nombre total de semaines entre date_debut et fin 2025
date_fin = pd.to_datetime('2025-12-31').normalize()
nb_semaines = ((date_fin - date_debut).days // 7) + 1

# Initialiser un tableau de CA hebdomadaire
ca_hebdo_cumule = np.zeros(nb_semaines)

# Créationd e ca_article
for ca_liste in df['ca_article']:
    for i, val in enumerate(ca_liste):
        if i < nb_semaines:
            ca_hebdo_cumule[i] += val

# Calculer le CA cumulé réel - addition ca_article /sem.
aujourdhui = pd.to_datetime(datetime.today().date())
semaine_actuelle = (aujourdhui - date_debut).days // 7
semaine_actuelle = min(semaine_actuelle, nb_semaines - 1)

ca_hebdo_reel = ca_hebdo_cumule[:semaine_actuelle+1]
ca_cumule_reel = np.cumsum(ca_hebdo_reel)

# Entraîner modèle de régression lin.
X_train = np.arange(semaine_actuelle + 1).reshape(-1, 1)
y_train = ca_cumule_reel


model = LinearRegression()
model.fit(X_train, y_train)

# Prédire le CA cumulé
X_pred = np.arange(nb_semaines).reshape(-1, 1)
y_pred = model.predict(X_pred)

dates_semaines = [date_debut + timedelta(weeks=i) for i in range(nb_semaines)]

fig = go.Figure()

#  CA cumulé actuel
fig.add_trace(go.Scatter(
    x=dates_semaines[:semaine_actuelle+1],
    y=ca_cumule_reel,
    mode='lines+markers',
    name='CA cumulé réel',
    line=dict(color='green')
))

# CA cumulé prédit
fig.add_trace(go.Scatter(
    x=dates_semaines,
    y=y_pred,
    mode='lines',
    name='CA cumulé prédit',
    line=dict(color='firebrick',)
))

# Sueprposition des traces et visualisation
fig.update_layout(
    title="Prédiction du CA cumulé - APDV",
    xaxis_title="Date",
    yaxis_title="Chiffre d'affaires cumulé (€)",
    template="plotly_white",
    hovermode="x unified"
)

fig.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Colab_Notebooks/AnalyseVentes_APDV.ipynb"')

