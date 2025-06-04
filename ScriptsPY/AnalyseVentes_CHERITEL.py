#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


get_ipython().system('pip install pdfplumber')


# In[ ]:


import os
import pdfplumber

factures_cheritel = '/content/drive/My Drive/Analyse des Ventes/Factures CHERITEL'

# Liste tous les fichiers PDF dans le dossier
fichiers_pdf = [
    f for f in os.listdir(factures_cheritel)
    if f.lower().endswith('.pdf')
]


# In[ ]:


import os
import pandas as pd
import pdfplumber

factures_cheritel = '/content/drive/My Drive/Analyse des Ventes/Factures CHERITEL'
csv_sortie = '/content/drive/My Drive/Analyse des Ventes/factures_cheritel.csv'

# Charger les factures déjà traitées s’il y en a
if os.path.exists(csv_sortie):
    df_total = pd.read_csv(csv_sortie)
    fichiers_deja_traitees = set(df_total['fichier'].unique())
else:
    df_total = pd.DataFrame()
    fichiers_deja_traitees = set()

# Lister les nouveaux fichiers PDF à traiter
fichiers_pdf = [
    f for f in os.listdir(factures_cheritel)
    if f.lower().endswith('.pdf') and f not in fichiers_deja_traitees
]

def extraire_infos_cheritel(texte):
    import re
    from datetime import datetime

    lignes = texte.split('\n')
    en_lecture = False
    date_facture = None
    articles = []

    for ligne in lignes:
        if "BL N°" in ligne and "DU" in ligne:
            try:
                date_str = ligne.strip().split()[-1]
                date_facture = datetime.strptime(date_str, "%d/%m/%Y").date()
            except Exception as e:
                print(f"Erreur de parsing de date dans la ligne : {ligne}\n{e}")
            break

    for ligne in lignes:
        if "BL N°" in ligne:
            en_lecture = True
            continue
        if "TOTAL BL" in ligne:
            break
        if not en_lecture:
            continue

        match = re.search(
            r"^(?P<designation>.+?)\s+(?P<colis>[\d,]+)\s+(?P<poids>[\d,]+)\s+(?P<unite>[A-Z]{1,3})\s+(?P<pu_ht>[\d,]+)\s+(?P<montant>[\d,]+)",
            ligne.strip()
        )
        if match:
            d = match.groupdict()
            try:
                colis = float(d["colis"].replace(",", "."))
                poids = float(d["poids"].replace(",", "."))
                pu_ht = float(d["pu_ht"].replace(",", "."))
                quantite = poids
                articles.append({
                    "designation": d["designation"].strip(),
                    "quantite_kg": quantite,
                    "prix_unitaire_ht": pu_ht,
                    "date": date_facture
                })
            except Exception as e:
                print(f"Erreur de conversion sur la ligne : {ligne}\n{e}")
                continue

    return pd.DataFrame(articles)


# Traiter chaque nouveau fichier PDF
for fichier in fichiers_pdf:
    chemin_complet = os.path.join(factures_cheritel, fichier)
    try:
        with pdfplumber.open(chemin_complet) as pdf:
            texte = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        df_facture = extraire_infos_cheritel(texte)
        if not df_facture.empty:
            df_facture['fichier'] = fichier
            df_total = pd.concat([df_total, df_facture], ignore_index=True)
            print(f"{fichier} → OK ({len(df_facture)} lignes)")
        else:
            print(f"{fichier} → Aucune donnée extraite.")
    except Exception as e:
        print(f"{fichier} → Erreur : {e}")

# Sauvegarde dans le CSV
df_total.to_csv(csv_sortie, index=False)
print(f"✅ Fichier mis à jour : {csv_sortie}")



# In[ ]:


display(df_total)


# ## Formatage des données

# In[ ]:


import pandas as pd

# S'assurer que la colonne "date" est bien en format datetime
df_total['date'] = pd.to_datetime(df_total['date'], errors='coerce')

df_total = df_total.dropna(subset=['date'])

# Trier par ordre croissant de la date
df_total_sorted = df_total.sort_values(by='date', ascending=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df_total_sorted)


# In[ ]:


#print(df_total.dtypes)
(df_total['quantite']*df_total['prix_unitaire_ht']).sum()


# In[ ]:


import pandas as pd


# Trier la DataFrame par 'designation' puis par 'date'
df_triee = df_total.sort_values(by=['designation', 'date'])

# Grouper par 'designation' et agréger les champs
df_groupee = df_triee.groupby('designation').agg(
    nb_commandes=('date', 'count'),
    quantites_commandees=('quantite', lambda x: list(x)),
    dates_commandes=('date', lambda x: list(x)),
    prix_unitaire_moyen=('prix_unitaire_ht', 'mean')
).reset_index()

df_groupee['prix_unitaire_moyen'] = df_groupee['prix_unitaire_moyen'].round(2)


# In[ ]:


#display(df_groupee.dtypes)


# In[ ]:


def estimer_ca_cheritel(row):
    quantites = row['quantites_commandees']
    pu = row['prix_unitaire_moyen']
    ca_total = 0
    for q in quantites:
        ca = q * pu * 1.3  # marge de 30%
        ca_total += ca
    return ca_total

df_groupee['ca_total_article'] = df_groupee.apply(estimer_ca_cheritel, axis=1)


# In[ ]:


print(df_groupee.head())


# In[ ]:


import pandas as pd

# Conversion explicite des dates en datetime
df_groupee['dates_commandes'] = df_groupee['dates_commandes'].apply(
    lambda lst: pd.to_datetime(lst, dayfirst=True, errors='coerce') if isinstance(lst, list) else []
)

print(df_groupee.head())

# Aplatir toutes les dates valides dans une seule liste
toutes_les_dates = [date for dates in df_groupee['dates_commandes'] for date in dates if pd.notna(date)]

# Vérifier qu'on a des dates valides
if toutes_les_dates:
    date_min = min(toutes_les_dates)
    date_max = max(toutes_les_dates)

    print("Première date de commande :", date_min.strftime("%d-%m-%Y"))
    print("Dernière date de commande :", date_max.strftime("%d-%m-%Y"))
else:
    print("Aucune date valide trouvée dans la colonne 'dates_commandes'.")




# In[ ]:


import pandas as pd
from datetime import datetime, timedelta

# Marge appliquée
MARGE = 1.3

# Normaliser toutes les dates au lundi de la semaine
def normalize_to_monday(d):
    return d - timedelta(days=d.weekday())

# Déterminer les bornes de l’intervalle d’étude
date_debut_globale = normalize_to_monday(min(df_groupee['dates_commandes'].apply(lambda lst: min(lst))))
date_fin_globale = normalize_to_monday(datetime.today())

# Calculer le nombre de semaines de l’intervalle
nb_semaines = ((date_fin_globale - date_debut_globale).days // 7) + 1

# Ajouter la colonne 'ca_semaine_article'
def calculer_ca_semaine_article(row):
    ca_par_semaine = [0.0] * nb_semaines
    prix = row['prix_unitaire_moyen']
    dates = row['dates_commandes']
    quantites = row['quantites_commandees']

    for j in range(len(dates)):
        date = normalize_to_monday(dates[j])
        index_semaine = (date - date_debut_globale).days // 7

        ca_etalé = (quantites[j] / 2) * prix * MARGE

        if 0 <= index_semaine < nb_semaines:
            ca_par_semaine[index_semaine] += ca_etalé
        if 0 <= index_semaine + 1 < nb_semaines:
            ca_par_semaine[index_semaine + 1] += ca_etalé

    return ca_par_semaine

df_groupee['ca_semaine_article'] = df_groupee.apply(calculer_ca_semaine_article, axis=1)


# In[ ]:


display(df_groupee.head())


# In[ ]:


from datetime import datetime, timedelta
import pandas as pd

# Fortement inspiré de la fonction 'etaler_ventes2' de 'AnalyseVentes_SUPERGROUP'
# Diff sur la période d'étalage due à la durée de vie des produits
def etaler_ventes_qte_par_semaine(ref):
    # Date de la première facture (globale)
    date_premiere_facture_globale = df_groupee['dates_commandes'].apply(min).min()
    today = datetime.today()
    lundi_courant = today - timedelta(days=today.weekday())
    nb_semaines = (lundi_courant - date_premiere_facture_globale).days // 7 + 1

    # Convertir les dates commandes en datetime
    commandes = pd.to_datetime(ref['dates_commandes'])
    quantites = ref['quantites_commandees']

    # DF temporaire
    df_temp = pd.DataFrame({'date_commande': commandes, 'quantite': quantites})

    # Calculer le lundi de la semaine pour chaque cmd
    df_temp['semaine'] = df_temp['date_commande'] - pd.to_timedelta(df_temp['date_commande'].dt.weekday, unit='d')

    # Regrouper par sem. et += les quantités
    df_grouped = df_temp.groupby('semaine')['quantite'].sum().reset_index()

    # Initialiser la liste des quantités par semaine
    quantites_par_semaine = [0.0] * nb_semaines

    for _, row in df_grouped.iterrows():
        semaine = row['semaine']
        qte = row['quantite']

        idx_semaine = (semaine - date_premiere_facture_globale).days // 7

        if 0 <= idx_semaine < nb_semaines:
            # Étaler la quantité sur 2 semaines : 50% cette semaine + 50% semaine suivante
            quantites_par_semaine[idx_semaine] += qte * 0.5
            if idx_semaine + 1 < nb_semaines:
                quantites_par_semaine[idx_semaine + 1] += qte * 0.5

    return quantites_par_semaine


# In[ ]:


def estimer_ca_par_semaine(row, marge=1.3):
    quantites_par_semaine = row['quantites_vendues']
    pu = row['prix_unitaire_moyen']

    ca_par_semaine = [q * pu * marge for q in quantites_par_semaine]
    return ca_par_semaine



# In[ ]:


df_groupee['quantites_vendues'] = df_groupee.apply(etaler_ventes_qte_par_semaine, axis=1)
df_groupee['ca_semaine_article'] = df_groupee.apply(estimer_ca_par_semaine, axis=1)


# In[ ]:


display(df_groupee.head())


# In[ ]:


# Total estimé
ca_estime = df_groupee['ca_total_article'].sum()

# Pour le CA réel : on somme les quantités par ligne avant multiplication
montant_ht = (df_groupee['quantites_commandees'].apply(sum) * df_groupee['prix_unitaire_moyen']).sum()

# Affichage
print(f"CA estimé total : {ca_estime:,.2f} €")
print(f"Montant HT   : {montant_ht:,.2f} €")
print(f"Marge HT   : {abs(ca_estime - montant_ht):,.2f} €")
print(f"Marge HT en %   : {100 * (ca_estime - montant_ht) / montant_ht:.2f} %")


# In[ ]:


import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Déterminer la date de départ (premier lundi de toutes les commandes)
date_premiere_facture = df_groupee['dates_commandes'].apply(min).min()
date_premiere_lundi = date_premiere_facture - timedelta(days=date_premiere_facture.weekday())

# Calculer le nombre de semaines dans les vecteurs
nb_semaines = len(df_groupee.iloc[0]['ca_semaine_article'])

# Générer la liste des dates (lundis)
dates_semaines = [date_premiere_lundi + timedelta(weeks=i) for i in range(nb_semaines)]

# Additionner le CA semaine par semaine
ca_total_par_semaine = [0.0] * nb_semaines
for liste_ca in df_groupee['ca_semaine_article']:
    for i, ca in enumerate(liste_ca):
        ca_total_par_semaine[i] += ca

# Créer un DataFrame pour le plot
df_ca = pd.DataFrame({
    'semaine': dates_semaines,
    'ca_total': ca_total_par_semaine
})

# Graphique interactif Plotly
fig = px.line(
    df_ca,
    x='semaine',
    y='ca_total',
    title='Chiffre d’affaires estimé par semaine (€)',
    labels={'semaine': 'Semaine', 'ca_total': 'CA hebdomadaire (€)'},
)

fig.update_traces(mode='lines+markers')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='CA par semaine (€)',
    hovermode='x unified'
)

fig.show()


# In[ ]:


import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Déterminer la date du premier lundi global
date_premiere_facture = df_groupee['dates_commandes'].apply(min).min()
date_premiere_lundi = date_premiere_facture - timedelta(days=date_premiere_facture.weekday())

# Nombre de semaines (longueur des vecteurs CA)
nb_semaines = len(df_groupee.iloc[0]['ca_semaine_article'])

# Générer la liste des lundis
dates_semaines = [date_premiere_lundi + timedelta(weeks=i) for i in range(nb_semaines)]

# Agrégation du CA par semaine
ca_total_par_semaine = [0.0] * nb_semaines
for liste_ca in df_groupee['ca_semaine_article']:
    for i, ca in enumerate(liste_ca):
        ca_total_par_semaine[i] += ca

# Calcul du CA cumulé
ca_cumule = pd.Series(ca_total_par_semaine).cumsum()

# DataFrame pour le graph
df_cumule = pd.DataFrame({
    'semaine': dates_semaines,
    'ca_cumule': ca_cumule
})

# Graphique Plotly
fig = px.line(
    df_cumule,
    x='semaine',
    y='ca_cumule',
    title='Chiffre d’affaires cumulé par semaine (€)',
    labels={'semaine': 'Semaine', 'ca_cumule': 'CA cumulé (€)'},
)

fig.update_traces(mode='lines+markers')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Chiffre d’affaires cumulé (€)',
    hovermode='x unified'
)

fig.show()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Préparation des données

X = np.arange(len(df_cumule)).reshape(-1, 1)
y = df_cumule['ca_cumule'].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression().fit(X_poly, y)

X_future = np.arange(len(df_cumule) + 20).reshape(-1, 1)
y_future = model.predict(poly.transform(X_future))


# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_holt = ExponentialSmoothing(df_cumule['ca_cumule'], trend='add', seasonal=None)
fit_holt = model_holt.fit()
forecast = fit_holt.forecast(20)


# In[ ]:


import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_cumule['semaine'], y=df_cumule['ca_cumule'], name='CA réel'))
fig.add_trace(go.Scatter(x=pd.date_range(start=df_cumule['semaine'].iloc[0], periods=len(X_future), freq="W-MON"), y=y_future, name='Poly deg2'))
fig.add_trace(go.Scatter(x=pd.date_range(start=df_cumule['semaine'].iloc[0], periods=len(df_cumule) + 20, freq="W-MON"), y=np.concatenate([df_cumule['ca_cumule'].values, forecast]), name='Holt'))

fig.update_layout(title='Prévision du CA cumulé', xaxis_title='Date', yaxis_title='CA cumulé (€)')
fig.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Colab_Notebooks/AnalyseVentes_CHERITEL.ipynb"')

