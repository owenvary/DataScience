#!/usr/bin/env python
# coding: utf-8

# #I. Import des données de vente

# ##1 - Import du drive contenant les factures

# In[147]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[148]:


#packages à installer
get_ipython().system('pip install pdfplumber')


# ###1.1 - Accès au dossier contenant les factures

# In[149]:


import os
import re
import pdfplumber
import pandas as pd


factures_supergroup = '/content/drive/My Drive/Analyse des Ventes/Factures SUPERGROUP'

# Liste tous les fichiers PDF dans le dossier
fichiers_pdf = [
    f for f in os.listdir(factures_supergroup)
    if f.lower().endswith('.pdf')
]

# Affiche les fichiers trouvés
print("📄 Factures trouvées :", fichiers_pdf)


# ##2 - Extraction des factures au format csv

# ###2.1 - Fonctions d'extraction des données pdf

# In[150]:


# Vérifie si une ligne correspond à un article en commençant par un code (6 à 13 chiffres), avec éventuellement un ">" devant
def ligne_est_article(ligne):
    return bool(re.match(r'^(>?)(\d{6,13})\s+', ligne.strip()))

# Convertit une chaîne de caractères en float, en gérant les formats français (virgule pour décimal, point pour milliers)
def convertir_vers_float(valeur_str):
    if valeur_str is None:
        return 0.0
    return float(valeur_str.replace('.', '').replace(',', '.'))

# Extrait les données d'un article à partir d'une ligne brute (code, désignation, quantités, prix, etc.)
def extraire_article_depuis_ligne(ligne):
    ligne = ligne.strip()

    # Extrait le code article (6 à 13 chiffres) au début de la ligne
    match_code = re.match(r'^(>?)(\d{6,13})\s+', ligne)
    if not match_code:
        return None  # Pas un article reconnu

    code = match_code.group(2)
    reste = ligne[match_code.end():]  # Le reste de la ligne, sans le code

    # Les 5 derniers blocs sont toujours : quantité, conditionnement, prix, montant, TVA
    blocs = reste.split()
    if len(blocs) < 5:
        return None  # Implique que la ligne a passé le filtre parasites mais n'est aps un article

    try:
        tva = convertir_vers_float(blocs[-1])
        montant_ht = convertir_vers_float(blocs[-2])
        prix_u_ht = convertir_vers_float(blocs[-3])
        cond = int(blocs[-4])
        quantite = int(blocs[-5])
    except ValueError:
        return None  # Une des valeurs numériques est mal formée

    # Tout ce qui reste avant les 5 blocs numériques est la désignation de l’article
    designation = " ".join(blocs[:-5])

    return [code, designation, quantite, cond, prix_u_ht, montant_ht, tva]

# Extrait tous les articles d’un bloc de texte (souvent une page de facture)
def extraire_articles_depuis_bloc(bloc_texte):
    lignes = bloc_texte.strip().split('\n')[2:]  # Ignore les deux premières lignes (souvent des en-têtes)
    articles = []
    articles_sans_famille = []

    # Lignes à ignorer car ce ne sont pas des articles
    parasites = [
        r'^dont Taxe', r'^Points acquis', r'^Code Client', r'^Représentant',
        r'^TOTAL HT', r'^page \d+/\d+', r'^>?\d{6,13}\s+GMS\s+',
        r'^SUPERGROUP', r'^SOROWEN',
    ]

    # Familles de produits possibles
    familles_possibles = [
        "CONFISERIE", "BOISSONS", "EPICERIE", "PATISSERIE",
        "GUM", "PIPIER", "BISCUITERIE", "BISCUITERIE SALEE"
    ]

    famille_en_attente = None  # Pour mémoriser la dernière famille détectée

    for ligne in lignes:
        l = ligne.strip()

        # Détecte si une ligne correspond à une famille de produits
        famille_detectee = None
        for famille in familles_possibles:
            if l.upper().startswith(famille):
                famille_detectee = famille.capitalize()
                break

        # Si une famille est détectée, on l’applique aux articles en attente
        if famille_detectee:
            famille_en_attente = famille_detectee
            for article in articles_sans_famille:
                article.append(famille_en_attente)
                articles.append(article)
            articles_sans_famille = []
            continue

        # Ignore les lignes parasites
        if any(re.match(p, l) for p in parasites):
            continue

        # Si la ligne est bien un article, on l'extrait
        if ligne_est_article(l):
            article = extraire_article_depuis_ligne(l)
            if article:
                articles_sans_famille.append(article)

    # À la fin, on assigne "Inconnue" aux articles sans famille
    for article in articles_sans_famille:
        article.append('Inconnue')
        articles.append(article)

    return articles

# Traite un PDF de facture en extrayant tous les articles présents, avec leur fournisseur et date
def traiter_facture_individuelle(chemin_pdf):
    all_articles = []

    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            texte = page.extract_text()
            if texte:
                articles = extraire_articles_depuis_bloc(texte)

                nom_fichier = os.path.basename(chemin_pdf)

                # Le nom du fournisseur est la partie avant "_Facture"
                fournisseur = nom_fichier.split('_Facture')[0]

                # On extrait la date au format AAAAMMJJ si elle est présente dans le nom du fichier
                match_date = re.search(r'_(\d{8})_', nom_fichier)
                if match_date:
                    date_facture = pd.to_datetime(match_date.group(1), format='%Y%m%d').date()
                else:
                    date_facture = None

                # On ajoute fournisseur et date à chaque ligne d'article
                for article in articles:
                    article.append(fournisseur)
                    article.append(date_facture)

                all_articles.extend(articles)

    # Si on a trouvé des articles, on les met dans un DataFrame
    if all_articles:
        df = pd.DataFrame(all_articles, columns=[
            "code", "designation", "quantite", "conditionnement",
            "prix_unitaire_ht", "montant_ht", "tva", "famille",
            "fournisseur", "date_facture"
        ])
        return df
    else:
        return pd.DataFrame()






# ###2.2 - Stockage des factures dans df_total

# In[151]:


# Chaque facture = 1 df puis concaténation
dfs = []

for nom_fichier in fichiers_pdf:
    chemin_pdf = os.path.join(factures_supergroup, nom_fichier)
    print(f"Traitement de : {nom_fichier}")
    df_temp_supergroup = traiter_facture_individuelle(chemin_pdf)
    dfs.append(df_temp_supergroup)

df_total = pd.concat(dfs, ignore_index=True)
df_total.to_csv(os.path.join(factures_supergroup, 'articles_supergroup.csv'), index=False)


# ###2.3 - Pré-visualisation de df_total

# In[152]:


chemin_csv = os.path.join(factures_supergroup, 'articles_supergroup.csv')
df_verification = pd.read_csv(chemin_csv)

# Affichage complet
display(df_verification)


# In[153]:


# Nb refs ?
print(f"Nb réf: {len(df_verification)}\n")

# Formats ?
print(f"Types: {df_verification.dtypes}")


# ##3 - Formattage des données

# ###3.1 - Formattage des dates (str -> datetime)

# In[154]:


# Format des dates
df_total['date_facture'] = pd.to_datetime(df_total['date_facture'], errors='coerce')  # Reconversion
df_total['date_facture'] = df_total['date_facture'].dt.strftime('%d/%m/%Y')  # Format JJ/MM/AAAA


# In[155]:


# Affichage des 5 premières lignes
display(df_total.head())


# In[156]:


# Répartition des articles par famille
repartition_refs = df_total['famille'].value_counts()
display(repartition_refs)

# Nombre d'articles mal classés (famille = 'Inconnue')
nb_inconnus = (df_total['famille'] == 'Inconnue').sum()

# Taux d'erreur = part des articles dont la famille n'a pas été détectée
taux_erreur = nb_inconnus / len(df_total)

print(f"Taux d'erreur de classification : {taux_erreur:.2%}\n")



# ###3.2 - Formattage des quantités et regroupement des lignes non uniques

# In[157]:


# Étape 1 : Quantité totale par ligne
df_total["quantite_totale_ligne"] = df_total["quantite"] * df_total["conditionnement"]

# Étape 2 : Tri pour identifier la dernière commande par article
df_total_sorted = df_total.sort_values("date_facture") # Ordre croissant

# Étape 3 : Dernière commande
df_last_command = (
    df_total_sorted.groupby("code").last().reset_index()
    [["code", "quantite", "conditionnement"]]
    .rename(columns={"quantite": "dernier_quantite", "conditionnement": "dernier_conditionnement"})
)

# Étape 4 : Synthèse par article (maj des variables suivantes)
df_synthese = (
    df_total.groupby(["code", "designation", "fournisseur", "famille"])
    .agg(
        quantite_totale=("quantite_totale_ligne", "sum"),
        quantite_commande_totale=("quantite", "sum"),
        conditionnement_moyen=("conditionnement", "mean"),
        prix_unitaire_moyen=("prix_unitaire_ht", "mean"),
        nb_commandes=("quantite", "count"),
        dates_commande=("date_facture", lambda x: sorted(set(x))),
    )
    .reset_index()
)

# Étape 5 : Fusion avec dernière commande en utilisant un left join
df_synthese = df_synthese.merge(df_last_command, on="code", how="left")

# Étape 6 : Calcul des bornes
df_synthese["quantite_vendue_min"] = df_synthese["quantite_totale"] - (
    df_synthese["dernier_quantite"] * df_synthese["dernier_conditionnement"]
)
df_synthese["quantite_vendue_max"] = df_synthese["quantite_totale"]

# Étape 7 : Format lisible de l'intervalle
df_synthese["quantite_vendue"] = df_synthese.apply(
    lambda row: f"{int(row.quantite_vendue_min)}"
    if row.quantite_vendue_min == row.quantite_vendue_max
    else f"{int(row.quantite_vendue_min)}-{int(row.quantite_vendue_max)}",
    axis=1
)

# Réorganisation finale
df_synthese = df_synthese[
    [
        "code", "designation", "fournisseur", "famille",
        "nb_commandes", "conditionnement_moyen",
        "quantite_totale", "quantite_vendue", "prix_unitaire_moyen",
        "nb_commandes", "dates_commande"
    ]
]


# In[158]:


# Aperçu
print(df_synthese.head())


# ###3.3 - Calcul du CA avant lissage des ventes

# In[159]:


# Nettoyer et extraire la borne min de quantite_vendue (format "min-max")
df_synthese["quantite_vendue_min"] = df_synthese["quantite_vendue"].str.extract(r"(\d+)", expand=False).astype(int)

# S'assurer que quantite_totale est bien numérique
df_synthese["quantite_totale"] = df_synthese["quantite_totale"].astype(float)

# Calculs
df_synthese["ca_potentiel"] = df_synthese["quantite_totale"] * df_synthese["prix_unitaire_moyen"]
df_synthese["ca_actuel"] = df_synthese["quantite_vendue_min"] * df_synthese["prix_unitaire_moyen"]
df_synthese["ca_potentiel_latent"] = df_synthese["ca_potentiel"] - df_synthese["ca_actuel"]




# In[160]:


display(df_synthese)


# ###3.4 - Visualisation des données principales

# In[161]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les colonnes numériques pertinentes (ajuste si besoin)
colonnes_numeriques = [
    "conditionnement_moyen",
    "quantite_totale",
    "prix_unitaire_moyen",
    "ca_potentiel",
    "ca_actuel",
    "ca_potentiel_latent"
]

# Initialiser la figure
n = len(colonnes_numeriques)
fig, axes = plt.subplots(nrows=(n + 1) // 2, ncols=2, figsize=(18, n * 2.5))
axes = axes.flatten()  # Pour un accès facile

# Tracer les histogrammes
for i, col in enumerate(colonnes_numeriques):
    if col in df_synthese.columns:
        # Convertir proprement en numérique, ignorer erreurs
        data = pd.to_numeric(df_synthese[col], errors='coerce')
        sns.histplot(data.dropna(), bins=50, kde=True, ax=axes[i], color="#69b3a2")
        axes[i].set_title(f"Distribution de {col}", fontsize=12)
    else:
        axes[i].text(0.5, 0.5, f"Colonne manquante : {col}", ha='center', va='center')
        axes[i].set_axis_off()

# Cacher les axes inutilisés si le nombre de graphes est impair
for j in range(i + 1, len(axes)):
    axes[j].set_axis_off()

plt.tight_layout()
plt.suptitle("Histogrammes des variables numériques de la synthèse", fontsize=16, y=1.02)
plt.show()

# Les courbes représente la densité de population


# ###3.5 - Visualisation du CA et de la répartition des articles (avant lissage)

# In[185]:


import matplotlib.pyplot as plt

# Définir la marge moyenne
marge_moyenne = 1.3

# Calcul du CA potentiel
df_synthese["ca_potentiel"] = df_synthese["quantite_totale"] * df_synthese["prix_unitaire_moyen"] * marge_moyenne

# Calcul du CA latent (différence entre le potentiel et le réel)
df_synthese["ca_latent"] = df_synthese["ca_potentiel"] - df_synthese["ca_actuel"]

# Agrégations par famille
nb_ref_par_famille = df_synthese.groupby("famille")["code"].nunique().sort_values(ascending=False)
volume_par_famille = df_synthese.groupby("famille")["quantite_totale"].sum().loc[nb_ref_par_famille.index]
ca_actuel_par_famille = df_synthese.groupby("famille")["ca_actuel"].sum().loc[nb_ref_par_famille.index]
ca_latent_par_famille = df_synthese.groupby("famille")["ca_latent"].sum().loc[nb_ref_par_famille.index]

# Création des subplots
fig, axs = plt.subplots(1, 3, figsize=(22, 6))

# Graph 1 : Nb de références
nb_ref_par_famille.plot(kind="bar", ax=axs[0], color="skyblue", edgecolor="black")
axs[0].set_title("Nb de références par famille", fontsize=14)
axs[0].set_ylabel("Références")
axs[0].tick_params(axis='x', rotation=45)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)

# Graph 2 : Volume total
volume_par_famille.plot(kind="bar", ax=axs[1], color="lightgreen", edgecolor="black")
axs[1].set_title("Volume total commandé par famille", fontsize=14)
axs[1].set_ylabel("Quantité")
axs[1].tick_params(axis='x', rotation=45)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)

# Graph 3 : CA actuel et latent superposés
axs[2].bar(ca_actuel_par_famille.index, ca_actuel_par_famille, label="CA Actuel", color="steelblue", edgecolor="black")
axs[2].bar(ca_latent_par_famille.index, ca_latent_par_famille, bottom=ca_actuel_par_famille, label="CA Latent", color="lightcoral", edgecolor="black")

axs[2].set_title("CA actuel vs potentiel par famille (avant lissage des ventes)", fontsize=14)
axs[2].set_ylabel("Montant (€)")
axs[2].tick_params(axis='x', rotation=45)
axs[2].grid(axis="y", linestyle="--", alpha=0.7)
axs[2].legend()

plt.tight_layout()
plt.show()


# In[163]:


# Vérification visuelle des valeurs
print(ca_latent_par_famille)
print(ca_actuel_par_famille)


# In[187]:


import pandas as pd

# Assurer que 'date_facture' est bien en datetime
df_total['date_facture'] = pd.to_datetime(df_total['date_facture'], errors='coerce')

# Calcul des quantités corrigées par le conditionnement
df_total['quantite_corrigee'] = df_total['quantite'] * df_total['conditionnement']

# Regroupement par référence
df_ref_info = df_total.groupby('code').agg(
    nb_commande=('code', 'size'),
    dates_commandes=('date_facture', lambda x: sorted(list(x))),
    quantites_commandees=('quantite_corrigee', lambda x: list(x)),
    prix_unitaire_moyen=('prix_unitaire_ht', 'mean'),
    famille=('famille', 'first')  # ou 'mode', ou 'unique' selon tes données
).reset_index()

#print(df_ref_info)



# ###3.6 - Définir l'intervalle de temps d'étude
# 

# In[188]:


from datetime import datetime

# Extraire toutes les dates dans une seule liste (format datetime)
toutes_les_dates = [date for dates in df_ref_info['dates_commandes'] for date in dates]

# Première commande
premiere_commande_globale = min(toutes_les_dates)

# Dernière commande
derniere_commande_globale = max(toutes_les_dates)

# Affichage formaté
print(f"Première commande : {premiere_commande_globale.strftime('%d-%m-%Y')}")
print(f"Dernière commande : {derniere_commande_globale.strftime('%d-%m-%Y')}")


# ###3.7 - Création de la df optimale d'étude

# On cherche à crée une df d'historique des ventes, pour cela on commence par créer une df contenant les infos clés de chaque référence unique.
# df_ref_info -> df_historique_ventes

# In[191]:


import pandas as pd
from datetime import datetime, timedelta

# Vérification du type et du nombre de valeurs NaT
print("Type de df_total['date_facture'] :", df_total['date_facture'].dtype)
print("Nombre de dates non converties (NaT) :", df_total['date_facture'].isna().sum())

# Création de la colonne semaine_annee (début de semaine) ---
df_total['semaine_annee'] = df_total['date_facture'].dt.to_period('W').apply(lambda r: r.start_time)

# Regroupement des données avec ajout de la désignation ---
df_ref_info = df_total.groupby('code').agg(
    designation=('designation', 'first'),
    famille=('famille', 'first'),
    nb_commande=('code', 'size'),
    quantites_commandees=('quantite_corrigee', lambda x: list(x)),
    dates_commandes=('date_facture', lambda x: list(x)),
    prix_unitaire_moyen=('prix_unitaire_ht', 'mean')
).reset_index()

# Conversion des dates au format texte ---
df_ref_info['dates_commandes'] = df_ref_info['dates_commandes'].apply(
    lambda x: [date.strftime('%Y-%m-%d') for date in x]
)

# Dates extrêmes ---
date_facture = df_ref_info['dates_commandes'].apply(max).max()
date_premiere_facture = df_ref_info['dates_commandes'].apply(min).min()

#print(date_premiere_facture)
print(df_ref_info)


# ###3.8 - Lissage des ventes

# Afin de lisser les ventes nous procédons de la façon suivante:
# 
# 1.   Si une référence est commandé à deux dates différentes, cela implique que le stock entier a été vendu. On étale donc les ventes proportionnellement selon le nb de semaines entre les deux dates.
# 2.   Si une référence n'a été commandée qu'une fois, on étale les ventes sur les 4 prochains mois (environ 17 semaines). *Il est rare de mettre plus de 4 mois à écouler un stock. Ces références correspondent à des offres limités ou à des produits arrêtés.

# In[167]:


from datetime import datetime, timedelta
import pandas as pd

def etaler_ventes(ref, date_premiere_facture_globale):
    # Convertir les dates au bon format
    commandes = [pd.to_datetime(d) for d in ref['dates_commandes']]
    quantites = ref['quantites_commandees']

    # Lundi de la semaine actuelle
    today = datetime.today()
    lundi_courant = today - timedelta(days=today.weekday())

    # Liste de tous les lundis entre date_premiere_facture_globale et aujourd’hui
    all_mondays = pd.date_range(start=date_premiere_facture_globale, end=lundi_courant, freq='W-MON')

    ventes = [0] * len(all_mondays)  # Liste initialisée à 0

    # Index de semaine pour la première commande de l'article
    for i in range(len(commandes)):
        date_debut = commandes[i]

        if i < len(commandes) - 1:
            date_fin = commandes[i + 1]
        else:
            # Étaler sur 17 semaines après la dernière commande
            date_fin = date_debut + timedelta(weeks=17)

        semaines_a_remplir = pd.date_range(start=date_debut, end=date_fin, freq='W-MON')
        qte_par_semaine = quantites[i] / len(semaines_a_remplir) if len(semaines_a_remplir) > 0 else 0

        # Remplir les bonnes positions dans la liste `ventes`
        for semaine in semaines_a_remplir:
            if semaine >= date_premiere_facture_globale and semaine <= lundi_courant:
                idx = (semaine - date_premiere_facture_globale).days // 7
                if idx < len(ventes):
                    ventes[idx] += qte_par_semaine

    return ventes


# In[194]:


# Date de début globale
date_premiere_facture_globale = df_total['date_facture'].min()
date_aujourd_hui = datetime.today()
lundi_courant = date_aujourd_hui - timedelta(days=date_aujourd_hui.weekday())

# Liste des semaines (lundi de chaque semaine)
all_mondays = pd.date_range(start=date_premiere_facture_globale, end=lundi_courant, freq='W-MON')

# Construction de la liste de dictionnaires
historique_ventes = []

for _, row in df_ref_info.iterrows():
    code = row['code']
    designation = row['designation']
    famille = row['famille']
    prix = row['prix_unitaire_moyen']
    nb_commande = row['nb_commande']
    quantites_commandees = row['quantites_commandees']
    dates_commandes = row['dates_commandes']

    # Liste des quantités hebdo étalées
    ventes_etalees = etaler_ventes(row, date_premiere_facture_globale)

    # Ajout d'une ligne par article, la colonne contient la liste des ventes
    historique_ventes.append({
        'code': code,
        'designation': designation,
        'famille': famille,
        'prix_unitaire_moyen': prix,
        'nb_commande': nb_commande,
        'quantites_commandees': quantites_commandees,
        'quantite_vendue': ventes_etalees,
        'dates_commandes': dates_commandes,
        'semaines': [d.strftime("%d/%m/%y") for d in all_mondays]
    })

# Création du DataFrame final
df_historique_ventes = pd.DataFrame(historique_ventes)

# Affichage d’un exemple
print(df_historique_ventes.head())


# ###3.9 - Visualisation des ventes

# In[195]:


import matplotlib.pyplot as plt

# Récupérer la liste des semaines (de 1ère commande à aujourd'hui)
semaines = df_historique_ventes.iloc[0]['semaines']

# Initialiser une liste à 0 pour chaque semaine
total_ventes_par_semaine = [0] * len(semaines)

# Additionner les ventes semaine par semaine
for ventes in df_historique_ventes['quantite_vendue']:
    total_ventes_par_semaine = [sum(x) for x in zip(total_ventes_par_semaine, ventes)]

# Tracer le graphique
plt.figure(figsize=(14, 6))
plt.plot(semaines, total_ventes_par_semaine, marker='o')
plt.xticks(rotation=45)
plt.title("Total des ventes par semaine (toutes références)")
plt.xlabel("Semaine")
plt.ylabel("Quantité vendue")
plt.tight_layout()
plt.grid(True)
plt.show()





# ###3.10 - Attribution du CA généré par semaine par référence

# In[196]:


# Définir la marge à 30%
marge = 1.3

# Calculer le CA hebdomadaire pour chaque ligne
df_historique_ventes['ca_article'] = df_historique_ventes.apply(
    lambda row: [q * row['prix_unitaire_moyen'] * marge for q in row['quantite_vendue']],
    axis=1
)



# In[197]:


# Recalcul du CA total (somme de tous les CA hebdomadaires)
ca_actuel = df_historique_ventes['ca_article'].apply(sum).sum()

print(f"Chiffre d'affaires actuel (avec marge {marge}): {ca_actuel:,.2f} €")

#print(df_historique_ventes[df_historique_ventes['famille'] == 'Epicerie'])


# ###3.11 - Visualisation du CA

# ####3.11.1 - CA global par semaine

# In[198]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Exploser les colonnes pour avoir une ligne par semaine par article
df_exploded = df_historique_ventes.explode(['semaines', 'ca_article'])

# Conversion en datetime
df_exploded['semaines'] = pd.to_datetime(df_exploded['semaines'], format='%d/%m/%y')

# Grouper par semaine
df_ventes_global = df_exploded.groupby('semaines')['ca_article'].sum().reset_index()

# Graphique global
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_ventes_global, x='semaines', y='ca_article', marker='o')
plt.title('Ventes globales par semaine (€)')
plt.ylabel('Chiffre d\'affaires (€)')
plt.xlabel('Semaine')
plt.grid(True)
plt.tight_layout()
plt.show()


# ####3.11.2 - CA par famille par semaine

# In[199]:


# Grouper par semaine et famille
df_famille = df_exploded.groupby(['semaines', 'famille'])['ca_article'].sum().reset_index()

# Graphique par famille
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_famille, x='semaines', y='ca_article', hue='famille', marker='o')
plt.title('Ventes par semaine et par famille (€)')
plt.ylabel('Chiffre d\'affaires (€)')
plt.xlabel('Mois')
plt.legend(title='Famille', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[200]:


import pandas as pd
import plotly.express as px

df_famille = df_exploded.groupby(['semaines', 'famille'])['ca_article'].sum().reset_index()

# Assurer le format datetime pour manipulation
df_famille['semaines'] = pd.to_datetime(df_famille['semaines'])

# Graph plotly (interactif), 1 subplot par famille
fig = px.line(
    df_famille,
    x='semaines',
    y='ca_article',
    color='famille',
    facet_col='famille',
    facet_col_wrap=3,
    markers=True,
    title="Ventes hebdomadaires par famille (€)",
    labels={
        'semaines': 'Semaine',
        'ca_article': 'CA (€)',
        'famille': 'Famille'
    }
)

# Params d'affichage
fig.update_xaxes(tickangle=45)
fig.update_layout(
    height=600,
    showlegend=False,
    title_font_size=18,
    margin=dict(t=60, b=40, l=40, r=40)
)

fig.show()


# ##4 - Prédictions

# ###4.1 - Prédiction du CA (toutes années)

# ####4.1.1 - Visualisation du CA cumulé

# In[202]:


# Grouper les données par semaine (global)
df_ventes_global = df_exploded.groupby('semaines')['ca_article'].sum().reset_index()
df_ventes_global['semaines'] = pd.to_datetime(df_ventes_global['semaines'])

# Ajouter le CA cumulé
df_ventes_global['ca_cumule'] = df_ventes_global['ca_article'].cumsum()

# Tracer
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df_ventes_global['semaines'], df_ventes_global['ca_cumule'], marker='o', linestyle='-')
plt.title('CA cumulé par semaine')
plt.xlabel('Semaine')
plt.ylabel('CA cumulé (€)')
plt.grid(True)
plt.tight_layout()
plt.show()


# ###4.1 - Création de la régression

# In[204]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Variable numérique pour la semaine (X)
df_ventes_global = df_ventes_global.sort_values('semaines')
df_ventes_global['num_semaine'] = (df_ventes_global['semaines'] - df_ventes_global['semaines'].min()).dt.days // 7

# 2. Préparer les données pour la régression
X = df_ventes_global[['num_semaine']]
y = df_ventes_global['ca_cumule']

# 3. Entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# 4. Prédire pour les semaines passées + futures (par ex. 35 semaines en plus)
max_week = df_ventes_global['num_semaine'].max()
week_numbers = np.arange(0, max_week + 36)
X_full = pd.DataFrame({'num_semaine': week_numbers})

y_pred = model.predict(X_full)

# 5. Générer les dates associées aux semaines
start_date = df_ventes_global['semaines'].min()
dates_full = start_date + pd.to_timedelta(X_full['num_semaine'] * 7, unit='D')


# In[207]:


import plotly.graph_objects as go

# Tracer la courbe du CA (pointillés, ocre)
scatter = go.Scatter(
    x=df_ventes_global['semaines'],
    y=y,
    mode='markers',
    name='CA cumulé réel',
    marker=dict(color='orange', size=6),
    hovertemplate='Date: %{x}<br>CA: %{y:.2f} €'
)

# Tracer la droite de régression (rouge, continue)
regression = go.Scatter(
    x=dates_full,
    y=y_pred,
    mode='lines',
    name='Régression linéaire',
    line=dict(color='red', width=2),
    hovertemplate='Date: %{x}<br>CA prédit: %{y:.2f} €'
)

# Figure complète
fig = go.Figure(data=[scatter, regression])
fig.update_layout(
    title='Régression linéaire sur le CA cumulé (SUPERGROUP)',
    xaxis_title='Date',
    yaxis_title='CA cumulé (€)',
    hovermode='x unified',
    template='plotly_white',
    width=1000,
    height=500
)
fig.show()

# Affichage de l’équation
a = model.coef_[0]
b = model.intercept_
print(f"Équation de la droite : CA_prédit = {a:.2f} × semaine + ({b:.2f})")


# In[208]:


display(df_ventes_global.head())


# In[210]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Calcul des métriques sur les données d'entraînement
y_pred_train = model.predict(X)
r2 = r2_score(y, y_pred_train)
mae = mean_absolute_error(y, y_pred_train)
rmse = np.sqrt(mean_squared_error(y, y_pred_train))

# Affichage clair
print("Évaluation du modèle de régression linéaire :")
print(f"R²    : {r2:.4f}")
print(f"MAE   : {mae:,.2f} €")
print(f"RMSE  : {rmse:,.2f} €")


# ###4.2 - Prédiction du CA(2025)

# ####4.2.1 - Prédiction du CA global

# In[212]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Créer la df d'entraînement
df_train = df_ventes_global.copy()
df_train['num_semaine'] = (df_train['semaines'] - df_train['semaines'].min()).dt.days // 7
df_train['ca_cumule'] = df_train['ca_article'].cumsum()

# Entraînement sur tout l’historique (pas que 2024)
X_train = df_train[['num_semaine']]
y_train = df_train['ca_cumule']

model = LinearRegression()
model.fit(X_train, y_train)

# Obtenir les semaines de 2025 (prédiction sur tout 2025)
start_2025 = pd.to_datetime('2025-01-01')
week_numbers_2025 = np.arange(0, 53)
week_offset = ((start_2025 - df_train['semaines'].min()).days) // 7
X_pred_2025 = pd.DataFrame({'num_semaine': week_numbers_2025 + week_offset})
y_pred_2025 = model.predict(X_pred_2025)

# Ajustement pour que le CA cumulé en 2025 commence à 0
y_pred_2025 = y_pred_2025 - y_pred_2025[0]
dates_pred_2025 = start_2025 + pd.to_timedelta(week_numbers_2025 * 7, unit='D')

# Extraire les données de 2025 pour les superposer à la régression
df_2025 = df_ventes_global[df_ventes_global['semaines'] >= '2025-01-01'].copy()
df_2025['ca_cumule'] = df_2025['ca_article'].cumsum()

# Tracer
scatter = go.Scatter(
    x=df_2025['semaines'],
    y=df_2025['ca_cumule'],
    mode='markers',
    name='CA cumulé réel (2025)',
    marker=dict(color='orange', size=6),
    hovertemplate='Date: %{x}<br>CA: %{y:.2f} €'
)

regression = go.Scatter(
    x=dates_pred_2025,
    y=y_pred_2025,
    mode='lines',
    name='Régression basée sur l’historique',
    line=dict(color='red', width=2),
    hovertemplate='Date: %{x}<br>CA prédit: %{y:.2f} €'
)

fig = go.Figure(data=[scatter, regression])
fig.update_layout(
    title='Prédiction du CA cumulé en 2025 (modèle entraîné toutes les ventes) - SUPERGROUP',
    xaxis_title='Date',
    yaxis_title='CA cumulé (€)',
    hovermode='x unified',
    template='plotly_white',
    width=1000,
    height=500
)
fig.show()


# In[213]:


# Évaluation de la régression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Prédire sur les données d'entraînement
y_pred_train = model.predict(X_train)

# Calcul des métriques
r2 = r2_score(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

# Affichage
print(f"R² : {r2:.4f}")
print(f"Erreur Absolue Moyenne : {mae:.2f} €")
print(f"Écart-type Moyen des Erreurs : {rmse:.2f} €")


# ####4.2.2 - Prédiction du CA par famille

# In[214]:


display(df_historique_ventes.head())


# In[236]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Création du df des ventes détaillées
rows = []
for _, row in df_historique_ventes.iterrows():
    code = row['code']
    designation = row['designation']
    famille = row['famille']
    prix = row['prix_unitaire_moyen']
    nb_commande = row['nb_commande']
    quantites_commandees = row['quantites_commandees']
    dates_commandes = row['dates_commandes']
    semaines = pd.to_datetime(row['semaines'], format='%d/%m/%y')
    ca_values = row['ca_article']

    for date, ca in zip(semaines, ca_values):
        rows.append({
            'code': code,
            'designation': designation,
            'famille': famille,
            'prix_unitaire_moyen': prix,
            'nb_commande': nb_commande,
            'quantites_commandees': quantites_commandees,
            'dates_commandes': dates_commandes,
            'semaines': date,
            'ca_article': ca
        })

df_ventes_detaillees = pd.DataFrame(rows)

# 2. Supprimer les zéros initiaux de chaque produit
df_ventes_detaillees = df_ventes_detaillees.sort_values(by=['code', 'semaines'])
df_ventes_detaillees = df_ventes_detaillees.groupby('code', group_keys=False).apply(
    lambda grp: grp.loc[grp['ca_article'].ne(0).idxmax():]
).reset_index(drop=True)

# 3. Ajouter le numéro de semaine relatif à la première date de la base
df_ventes_detaillees['ca_article'] = df_ventes_detaillees['ca_article'].astype(float)
df_ventes_detaillees['num_semaine'] = (
    (df_ventes_detaillees['semaines'] - df_ventes_detaillees['semaines'].min()).dt.days // 7
)

# 4. Régression et stockage des résultats par famille
figs = []
start_2025 = pd.to_datetime('2025-01-01')
num_semaine_2025 = np.arange(0, 53)  # Les semaines de 2025

for famille, group in df_ventes_detaillees.groupby('famille'):
    df_ventes_famille = group.groupby('semaines')['ca_article'].sum().reset_index()
    df_ventes_famille = df_famille.sort_values('semaines')
    df_ventes_famille['num_semaine'] = (
        (df_ventes_famille['semaines'] - df_ventes_detaillees['semaines'].min()).dt.days // 7
    )
    df_ventes_famille['ca_cumule'] = df_ventes_famille['ca_article'].cumsum()

    # Trouver la première commande réelle de la famille
    min_date = df_ventes_famille[df_famille['ca_article'] > 0]['semaines'].min()

    # Déterminer la date de début de prédiction
    start_pred_date = max(start_2025, min_date)

    # Calculer la semaine de départ
    semaine_offset = ((start_pred_date - df_ventes_famille['semaines'].min()).days) // 7

    # Régression
    X = df_ventes_famille[['num_semaine']]
    y = df_ventes_famille['ca_cumule']
    model = LinearRegression()
    model.fit(X, y)

    # Prédiction 2025
    X_pred = pd.DataFrame({'num_semaine': num_semaine_2025 + semaine_offset})
    y_pred = model.predict(X_pred)
    y_pred = y_pred - y_pred[0]  # Ajustement à 0 à partir de la date de première commande

    # df avec les données 2025 (uniquement): pour graphs
    df_2025 = df_ventes_famille[df_ventes_famille['semaines'] >= start_2025].copy()
    df_2025['ca_cumule'] = df_2025['ca_article'].cumsum()

    # Évaluation
    y_pred_eval = model.predict(X)
    r2 = r2_score(y, y_pred_eval)
    mae = mean_absolute_error(y, y_pred_eval)
    rmse = np.sqrt(mean_squared_error(y, y_pred_eval))

    # Tracé
    fig = go.Figure()

    # Ventes réelles
    fig.add_trace(go.Scatter(
        x=df_2025['semaines'],
        y=df_2025['ca_cumule'],
        mode='markers',
        name='CA réel',
        marker=dict(color='orange', size=6),
        hovertemplate='Date: %{x}<br>CA: %{y:.2f} €'
    ))

    # Régression prévisionnelle 2025
    fig.add_trace(go.Scatter(
        x=start_pred_date + pd.to_timedelta(num_semaine_2025 * 7, unit='D'),
        y=y_pred,
        mode='lines',
        name='Régression',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>CA prédit: %{y:.2f} €'
    ))

    fig.update_layout(
        title=f"{famille} – (R²: {r2:.3f}, MAE: {mae:.2f} €, RMSE: {rmse:.2f} €)",
        xaxis_title="Date",
        yaxis_title="CA cumulé (€)",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=400
    )
    figs.append(fig)

# Affichage des graphiques
for fig in figs:
    fig.show()


# In[237]:


# Créer un DataFrame avec les meilleures réf/famille
top_references_by_famille = []

# Groupement par famille et tri des références
for famille, group in df_ventes_detaillees.groupby('famille'):

    df_famille_group = group.groupby('code').agg({
        'ca_article': 'sum',
        'designation': 'first',
        'prix_unitaire_moyen': 'first',
        'nb_commande': 'sum',
        'quantites_commandees': 'sum',
        'dates_commandes': 'first',
    }).reset_index()

    # Trier par CA décroissant et prendre les 10 meilleures références
    top_references = df_famille_group.nlargest(10, 'ca_article')

    top_references['famille'] = famille
    top_references_by_famille.append(top_references)

# Fusionner les résultats
df_top_references = pd.concat(top_references_by_famille, ignore_index=True)

# Afficher les 10 meilleures références pour chaque famille
print(df_top_references)


# #Exportation Git

# In[244]:


get_ipython().system('apt-get install git -y')


# In[245]:


get_ipython().system('git config --global user.name "owenvary"')
get_ipython().system('git config --global user.email "owen.vary@isen-ouest.yncrea.fr"')


# In[246]:


get_ipython().run_line_magic('cd', '/content')
get_ipython().system('git init')


# In[247]:


get_ipython().system('jupyter nbconvert --to script "/content/drive/MyDrive/Colab Notebooks/AnalyseVentes.ipynb"')


# In[248]:


get_ipython().system('cp "/content/drive/MyDrive/Colab Notebooks/AnalyseVentes.py" "/content/AnalyseVentes.py"')


# In[249]:


get_ipython().system('ls "/content/drive/MyDrive/Colab Notebooks/"')


# In[ ]:




