#!/usr/bin/env python
# coding: utf-8

# ##1 - Import du drive contenant les factures

# In[8]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


#packages à installer
get_ipython().system('pip install pdfplumber')


# ###1.1 - Accès au dossier contenant les factures

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


chemin_csv = os.path.join(factures_supergroup, 'articles_supergroup.csv')
df_verification = pd.read_csv(chemin_csv)

# Affichage complet
display(df_verification)


# In[ ]:


# Nb refs ?
print(f"Nb réf: {len(df_verification)}\n")

# Formats ?
print(f"Types: {df_verification.dtypes}")


# ##3 - Formattage des données

# ###3.1 - Formattage des dates (str -> datetime)

# In[ ]:


# Format des dates
df_total['date_facture'] = pd.to_datetime(df_total['date_facture'], errors='coerce')  # Reconversion
df_total['date_facture'] = df_total['date_facture'].dt.strftime('%d/%m/%Y')  # Format JJ/MM/AAAA


# In[ ]:


# Affichage des 5 premières lignes
display(df_total.head())


# In[ ]:


# Répartition des articles par famille
repartition_refs = df_total['famille'].value_counts()
display(repartition_refs)

# Nombre d'articles mal classés (famille = 'Inconnue')
nb_inconnus = (df_total['famille'] == 'Inconnue').sum()

# Taux d'erreur = part des articles dont la famille n'a pas été détectée
taux_erreur = nb_inconnus / len(df_total)

print(f"Taux d'erreur de classification : {taux_erreur:.2%}\n")



# ###3.2 - Formattage des quantités et regroupement des lignes non uniques

# In[ ]:


def attribuer_famille_connue(group):
    familles_connues = group[group['famille'] != 'Inconnue']['famille']
    if not familles_connues.empty:
        famille_majoritaire = familles_connues.mode().iloc[0]
        group['famille'] = famille_majoritaire
    return group

df_total['famille'] = df_total['famille'].fillna('Inconnue')  # sécurité
df_total = df_total.groupby('code', group_keys=False).apply(attribuer_famille_connue).reset_index(drop=True)


# In[ ]:


# Répartition des articles par famille
repartition_refs = df_total['famille'].value_counts()
display(repartition_refs)

# Nombre d'articles mal classés (famille = 'Inconnue')
nb_inconnus = (df_total['famille'] == 'Inconnue').sum()

# Taux d'erreur = part des articles dont la famille n'a pas été détectée
taux_erreur = nb_inconnus / len(df_total)

print(f"Taux d'erreur de classification : {taux_erreur:.2%}\n")


# In[ ]:


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


# In[ ]:


# Aperçu
print(df_synthese.head())


pd.set_option('display.max_rows', None)
display(df_synthese)
pd.reset_option('display.max_rows')


# ###3.3 - Calcul du CA avant lissage des ventes

# In[ ]:


# Nettoyer et extraire la borne min de quantite_vendue (format "min-max")
df_synthese["quantite_vendue_min"] = df_synthese["quantite_vendue"].str.extract(r"(\d+)", expand=False).astype(int)

# S'assurer que quantite_totale est bien numérique
df_synthese["quantite_totale"] = df_synthese["quantite_totale"].astype(float)

# Calculs
df_synthese["ca_potentiel"] = df_synthese["quantite_totale"] * df_synthese["prix_unitaire_moyen"]
df_synthese["ca_actuel"] = df_synthese["quantite_vendue_min"] * df_synthese["prix_unitaire_moyen"]
df_synthese["ca_potentiel_latent"] = df_synthese["ca_potentiel"] - df_synthese["ca_actuel"]




# In[ ]:


display(df_synthese)


# ###3.4 - Visualisation des données principales

# In[ ]:


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

# In[ ]:


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


# In[ ]:


# Vérification visuelle des valeurs
print(ca_latent_par_famille)
print(ca_actuel_par_famille)


# In[ ]:


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

# In[ ]:


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

# In[ ]:


import pandas as pd
from datetime import datetime, timedelta

# Vérification du type et du nombre de valeurs NaT
print("Type de df_total['date_facture'] :", df_total['date_facture'].dtype)
print("Nombre de dates non converties (NaT) :", df_total['date_facture'].isna().sum())

# On filtre les lignes sans date
df_total_clean = df_total.dropna(subset=['date_facture']).copy()

# Création d'une colonne pour le début de semaine
df_total_clean['semaine_annee'] = df_total_clean['date_facture'].dt.to_period('W').apply(lambda r: r.start_time)

print("Type de df_total['date_facture'] :", df_total_clean['date_facture'].dtype)
print("Nombre de dates non converties (NaT) :", df_total_clean['date_facture'].isna().sum())

# Regroupement des données avec ajout de la désignation
df_ref_info = df_total_clean.groupby('code').agg(
    designation=('designation', 'first'),
    famille=('famille', 'first'),
    nb_commande=('code', 'size'),
    quantites_commandees=('quantite_corrigee', lambda x: list(x)),
    dates_commandes=('date_facture', lambda x: list(x)),
    prix_unitaire_moyen=('prix_unitaire_ht', 'mean')
).reset_index()

# Conversion des dates au format texte
df_ref_info['dates_commandes'] = df_ref_info['dates_commandes'].apply(
    lambda x: [date.strftime('%Y-%m-%d') for date in x]
)

# Dates extrêmes
date_derniere_facture = df_ref_info['dates_commandes'].apply(max).max()
date_premiere_facture = df_ref_info['dates_commandes'].apply(min).min()

print("Date de la première facture :", date_premiere_facture)
print("Date de la dernière facture :", date_derniere_facture)
print(df_ref_info)


# ###3.8 - Lissage des ventes

# Afin de lisser les ventes nous procédons de la façon suivante:
# 
# 1.   Si une référence est commandé à deux dates différentes, cela implique que le stock entier a été vendu. On étale donc les ventes proportionnellement selon le nb de semaines entre les deux dates.
# 2.   Si une référence n'a été commandée qu'une fois, on étale les ventes sur les 4 prochains mois (environ 17 semaines). *Il est rare de mettre plus de 4 mois à écouler un stock. Ces références correspondent à des offres limités ou à des produits arrêtés.
# *Défaut : chaque commande reset le stock ce qui est vrai en théorie

# In[ ]:


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


# In[ ]:


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

# In[ ]:


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

# In[ ]:


# Définir la marge à 30%
marge = 1.3

# Calculer le CA hebdomadaire pour chaque ligne
df_historique_ventes['ca_article'] = df_historique_ventes.apply(
    lambda row: [q * row['prix_unitaire_moyen'] * marge for q in row['quantite_vendue']],
    axis=1
)



# In[ ]:


# Recalcul du CA total (somme de tous les CA hebdomadaires)
ca_actuel = df_historique_ventes['ca_article'].apply(sum).sum()

print(f"Chiffre d'affaires actuel (avec marge {marge}): {ca_actuel:,.2f} €")

#print(df_historique_ventes[df_historique_ventes['famille'] == 'Epicerie'])


# ###3.11 - Visualisation du CA

# ####3.11.1 - CA global par semaine

# In[ ]:


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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


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

# In[ ]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Variable numérique pour la semaine (X)
df_ventes_global = df_ventes_global.sort_values('semaines')
df_ventes_global['num_semaine'] = (df_ventes_global['semaines'] - df_ventes_global['semaines'].min()).dt.days // 7

# Préparer les données pour la régression
X = df_ventes_global[['num_semaine']]
y = df_ventes_global['ca_cumule']

# Entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# Prédire pour les semaines passées + futures (par ex. 35 semaines en plus)
max_week = df_ventes_global['num_semaine'].max()
week_numbers = np.arange(0, max_week + 36)
X_full = pd.DataFrame({'num_semaine': week_numbers})

y_pred = model.predict(X_full)

# Générer les dates associées aux semaines
start_date = df_ventes_global['semaines'].min()
dates_full = start_date + pd.to_timedelta(X_full['num_semaine'] * 7, unit='D')


# In[ ]:


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


# In[ ]:


display(df_ventes_global.head())


# In[ ]:


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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


display(df_historique_ventes)


# In[ ]:


display(df_synthese.head())


# In[ ]:


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
    quantites_vendues = row['quantite_vendue']

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
            'quantite_vendue': quantites_vendues,
            'ca_article': ca
        })

df_ventes_detaillees = pd.DataFrame(rows)

# Supprimer les zéros initiaux par produit
df_ventes_detaillees = df_ventes_detaillees.sort_values(by=['code', 'semaines'])
df_ventes_detaillees = df_ventes_detaillees.groupby('code', group_keys=False).apply(
    lambda grp: grp.loc[grp['ca_article'].ne(0).idxmax():]
).reset_index(drop=True)

# Convertir dates et calculer num_semaine relatif
df_ventes_detaillees['semaines'] = pd.to_datetime(df_ventes_detaillees['semaines'])
df_ventes_detaillees['ca_article'] = df_ventes_detaillees['ca_article'].astype(float)

date_min = df_ventes_detaillees['semaines'].min()
df_ventes_detaillees['num_semaine'] = ((df_ventes_detaillees['semaines'] - date_min).dt.days // 7)

start_2025 = pd.to_datetime('2025-01-01')
num_semaine_2025 = np.arange(0, 53)  # 53 semaines en 2025

figs = []

for famille, group in df_ventes_detaillees.groupby('famille'):
    # Somme du CA par semaine
    df_ventes_famille = group.groupby('semaines')['ca_article'].sum().reset_index()
    df_ventes_famille = df_ventes_famille.sort_values('semaines')
    df_ventes_famille['num_semaine'] = ((df_ventes_famille['semaines'] - date_min).dt.days // 7)
    df_ventes_famille['ca_cumule'] = df_ventes_famille['ca_article'].cumsum()

    # Date de première commande réelle dans la famille (CA > 0)
    min_date = df_ventes_famille[df_ventes_famille['ca_article'] > 0]['semaines'].min()
    if pd.isna(min_date):
        print(f"Aucune commande réelle pour la famille {famille}, saut.")
        continue

    # Date de début des prédictions
    start_pred_date = max(start_2025, min_date)
    semaine_offset = ((start_pred_date - date_min).days) // 7

    # Préparation régression
    X = df_ventes_famille[['num_semaine']]
    y = df_ventes_famille['ca_cumule']
    model = LinearRegression()
    model.fit(X, y)

    # Prédictions pour 2025
    X_pred = pd.DataFrame({'num_semaine': num_semaine_2025 + semaine_offset})
    y_pred = model.predict(X_pred)
    y_pred = y_pred - y_pred[0]  # Ajuster à 0 à la date de première commande

    # Données réelles 2025 (pour affichage)
    df_2025 = df_ventes_famille[df_ventes_famille['semaines'] >= start_2025].copy()
    df_2025['ca_cumule'] = df_2025['ca_article'].cumsum()

    # Évaluation sur données connues
    y_pred_eval = model.predict(X)
    r2 = r2_score(y, y_pred_eval)
    mae = mean_absolute_error(y, y_pred_eval)
    rmse = np.sqrt(mean_squared_error(y, y_pred_eval))

    # Trace Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_2025['semaines'],
        y=df_2025['ca_cumule'],
        mode='markers',
        name='CA réel',
        marker=dict(color='orange', size=6),
        hovertemplate='Date: %{x}<br>CA: %{y:.2f} €'
    ))

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

# Affichage des graphs
for fig in figs:
    fig.show()



# On remarque que les régressions qui manquent de précision sont celles concernant les familles 'récentes' impliquant un manque de données.

# In[ ]:


#display(df_ventes_detaillees.head())


# ###5 - Étude des meilleurs références SUPERGROUP

# ####5.1 - Création du df_top_references

# In[ ]:


# Créer un DataFrame avec les meilleures réf/famille
top_references_by_famille = []

# Groupement par famille et tri des références
for famille, group in df_ventes_detaillees.groupby('famille'):

    df_famille_group = group.groupby('code').agg({
        'ca_article': 'sum',
        'designation': 'first',
        'prix_unitaire_moyen': 'first',
        'nb_commande': 'first',
        'quantites_commandees': 'first',
        'quantite_vendue': 'first',
        'dates_commandes': 'first',
    }).reset_index()

    # Trier par CA décroissant et prendre les 10 meilleures références
    top_references = df_famille_group.nlargest(10, 'ca_article')

    top_references['famille'] = famille
    top_references_by_famille.append(top_references)

# Fusionner les rés.
df_top_references = pd.concat(top_references_by_famille, ignore_index=True)

# Afficher les 10 meilleures réfs par famille
display(df_top_references)


# ####5.2 - Visualisation des meilleurs réfs par famille

# *Graphs interactifs, mais difficile à lire dans l'état due à la quantité d'information. Possibilité de déselectionner les références en cliquant sur la légende associée

# #####5.2.1 - Visualisation des ventes par refs par fam.

# In[ ]:


import pandas as pd
import plotly.graph_objects as go

# Copies pour éviter de modifier les df d'origine
top_codes = df_top_references['code'].unique()
df_top_ventes = df_ventes_detaillees[df_ventes_detaillees['code'].isin(top_codes)].copy()

# Suppr. les semaines sans ventes :(avant 1ère cmd) ajuste le poitn de départ du graph et évite de biaiser la données d'entraînement pour la suite
df_top_ventes = df_top_ventes.sort_values(by=['code', 'semaines'])
df_top_ventes = df_top_ventes.groupby('code', group_keys=False).apply(
    lambda grp: grp.loc[grp['ca_article'].ne(0).idxmax():]
).reset_index(drop=True)

# Boucle sur les refs par famille
for famille, df_famille in df_top_ventes.groupby('famille'):
    fig = go.Figure()

    for code, df_produit in df_famille.groupby('code'):
        designation = df_produit['designation'].iloc[0]
        semaines = df_produit['semaines']
        ca_values = df_produit['ca_article']

        # Tracer la courbe des ventes
        fig.add_trace(go.Scatter(
            x=semaines,
            y=ca_values,
            mode='lines+markers',
            name=designation,
            marker=dict(symbol='circle', size=6),
            line=dict(width=2),
            hovertemplate='Semaine: %{x|%d/%m/%Y}<br>CA: %{y:.2f} €'
        ))

    # Maj de la mise en page
    fig.update_layout(
        title=f"Ventes hebdomadaires – {famille} (Top 10 produits)",
        xaxis_title="Semaine",
        yaxis_title="Chiffre d'affaires (€)",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=500
    )

    fig.show()


# #####5.2.2 - Visualisation des qté commandées par refs par fam.

# In[ ]:


import pandas as pd
import plotly.graph_objects as go

#  f(x) qui normalise ttes les dates au lundi correspondant
def normalize_to_monday(d):
    return d - pd.Timedelta(days=d.weekday())

# Boucle sur chaque top_refs par famille
for famille, df_famille in df_top_references.groupby('famille'):
    fig = go.Figure()

    for _, row in df_famille.iterrows():
        code = row['code']
        designation = row['designation']
        dates_raw = pd.to_datetime(row['dates_commandes'])  # format dt?
        quantites_raw = row['quantites_commandees']

        # Normalisation des dates de cmd
        dates_norm = [normalize_to_monday(d) for d in dates_raw]

        # Association date : quantité commandée
        quantites_par_date = {d: q for d, q in zip(dates_norm, quantites_raw)}

        # Génération de toutes les semaines entre la 1ère et la dernière cmd (=intervalle d'étude)
        toutes_semaines = pd.date_range(start=min(dates_norm), end=max(dates_norm), freq='W-MON')

        # Création d'une liste de quantités alignée sur toutes les semaines (0 si pas de commande cette semaine)
        # Permet que toutes les listes fassent la mm taille pour la création des graphs
        quantites_finales = [quantites_par_date.get(semaine, 0) for semaine in toutes_semaines]

        # Ajoute les tracés au mm graph
        fig.add_trace(go.Scatter(
            x=toutes_semaines,
            y=quantites_finales,
            mode='lines+markers',
            name=designation,
            marker=dict(symbol='circle', size=6),
            line=dict(width=2),
            hovertemplate='Semaine: %{x|%d/%m/%Y}<br>Quantité: %{y}'
        ))

    fig.update_layout(
        title=f"Quantités commandées – {famille} (Top 10 produits)",
        xaxis_title="Semaine",
        yaxis_title="Quantités commandées",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=500
    )

    fig.show()



# ###5.3 - Prédiction des quantités à commander selon la fréquence de commande

# Le modèle cherche à prédire les quantités sur les 4 semaines qui suivent la dernière commande en analysant la fréquence et les patterns de commande. Ce modèle est cohérent mais peut facilement être biaisé s'il y a des problèmes de commande par exemple.

# In[ ]:


import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 1 - Préparation des données
# Copie pour ne pas modifier l'original
df_top_references_pred = df_top_references.copy()

# Formats ?
df_top_references_pred['dates_commandes'] = df_top_references_pred['dates_commandes'].apply(
    lambda x: pd.to_datetime(x) if isinstance(x, list) else pd.to_datetime(eval(x)) #eval convertit des chaînes contenant des lsites
)
df_top_references_pred['quantites_commandees'] = df_top_references_pred['quantites_commandees'].apply(
    lambda x: x if isinstance(x, list) else eval(x)
)

# Récupérer les conditionnements depuis df_synthese
df_conditionnements = df_synthese[['code', 'conditionnement_moyen']]
df_top_references_pred = df_top_references_pred.merge(df_conditionnements, on='code', how='left')

# Normalisation des dates et quantités
def normalize_to_monday(d):
    return d - pd.Timedelta(days=d.weekday())

def normalize_quantities(dates_commandes, quantites, debut, fin):
    dates_norm = [normalize_to_monday(pd.to_datetime(d)) for d in dates_commandes]
    toutes_semaines = pd.date_range(start=debut, end=fin, freq='W-MON')
    mapping = dict(zip(dates_norm, quantites))
    return [mapping.get(semaine, 0) for semaine in toutes_semaines], toutes_semaines

donnees_produits = []

for _, row in df_top_references_pred.iterrows():
    code = row['code']
    dates = row['dates_commandes']
    quantites = row['quantites_commandees']
    cond = row['conditionnement_moyen']
    debut = normalize_to_monday(min(pd.to_datetime(dates)))
    fin = normalize_to_monday(pd.Timestamp.today())
    qte_norm, semaines = normalize_quantities(dates, quantites, debut, fin)
    donnees_produits.append({
        'code': code,
        'semaines': semaines,
        'quantites': qte_norm,
        'conditionnement_moyen': cond
    })

# 2 - Analyse & prédiction
resultats_pred = []

for produit in donnees_produits:
    code = produit['code']
    semaines = produit['semaines']
    quantites = produit['quantites']
    cond = produit['conditionnement_moyen']

    serie = pd.Series(quantites, index=semaines)

    # Calcul de la fréquence moyenne entre les commandes
    dates_commandes = serie[serie > 0].index
    if len(dates_commandes) < 2:
        continue  # pas assez de données

    deltas = dates_commandes.to_series().diff().dropna().dt.days
    frequence_moyenne = int(round(deltas.mean()))

    # Calcul d'une tendance linéaire
    x = np.arange(len(serie))
    y = np.array(serie)
    coeffs = np.polyfit(x, y, deg=1)
    tendance = np.poly1d(coeffs)

    # Prédiction pour les 4 prochaines semaines
    last_date = serie.index[-1]
    prochaines_semaines = [last_date + pd.Timedelta(weeks=i) for i in range(1, 5)]

    for i, semaine in enumerate(prochaines_semaines):
        jours_depuis_derniere_commande = (semaine - dates_commandes[-1]).days
        prob_commande = jours_depuis_derniere_commande >= frequence_moyenne

        if prob_commande:
            quantite_predite = max(0, int(round(tendance(len(serie) + i) / cond)) * cond)
        else:
            quantite_predite = 0

        resultats_pred.append({
            'code': code,
            'date_prevue': semaine,
            'quantite_predite': quantite_predite
        })

# Résultats
df_predictions = pd.DataFrame(resultats_pred)

df_predictions = df_predictions.merge(
    df_top_references_pred[['code', 'designation', 'famille']].drop_duplicates(),
    on='code', how='left'
)


# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px

# Fonction pour détecter une périodicité dans les dernières semaines d'une série
def detecter_periodicite_recente(series_quantites, max_lag=20, window=16):
    # On ne regarde que les x dernières semaines
    serie_recente = series_quantites[-window:]

    # Calcul des autocorrélations pour plusieurs décalages de semaines (=lag)
    autocorrs = [serie_recente.autocorr(lag=lag) for lag in range(1, max_lag+1)]

    # Si tous les résultats sont NaN : impraticable
    if all(np.isnan(autocorrs)):
        return None

    # Retourne le lag avec la meilleure corrélation (indice + 1 car init. à 0)
    period = np.nanargmax(autocorrs) + 1
    return period

def predire_par_repetition(dates_historique, quantites_historique, n_semaines_pred=8, conditionnement=1):
    """
    Prédit les prochaines commandes en répétant le schéma des semaines passées.
    La prédiction est arrondie au multiple du conditionnement.
    """

    # Sous-fonction pour aligner les dates sur les lundis
    def normalize_to_monday(d):
        return d - pd.Timedelta(days=d.weekday())

    # Normalise les dates de l'historique des ventes
    dates_norm = [normalize_to_monday(pd.to_datetime(d)) for d in dates_historique]
    debut = min(dates_norm)
    fin = normalize_to_monday(pd.Timestamp.today())

    # Crée la liste complète des semaines
    toutes_semaines = pd.date_range(start=debut, end=fin, freq='W-MON')
    mapping = dict(zip(dates_norm, quantites_historique))
    quantites_norm = [mapping.get(sem, 0) for sem in toutes_semaines]

    # Série temporelle complète (index = lundis, valeurs = quantités)
    serie = pd.Series(quantites_norm, index=toutes_semaines)

    # Détecte la périodicité de la série
    periode = detecter_periodicite_recente(serie)
    if periode is None:
        print("Périodicité non détectée, utilisation d'une périodicité de 1 semaine par défaut.")
        periode = 1
    else:
        print(f"Périodicité détectée : {periode} semaines.")

    # Génère les prochaines semaines à prédire
    dernier_lundi = toutes_semaines[-1]
    dates_futures = [dernier_lundi + pd.Timedelta(weeks=i) for i in range(1, n_semaines_pred+1)]

    # Répète la dernière séquence connue selon la périodicité détectée
    seq_periodique = quantites_norm[-periode:]
    predictions_brutes = []
    for i in range(n_semaines_pred):
        val = seq_periodique[i % periode]
        # Arrondit à un multiple du conditionnement
        val_arrondi = int(round(val / conditionnement) * conditionnement)
        predictions_brutes.append(max(0, val_arrondi))  # jamais de prédiction négative

    # Format final des prédictions
    df_pred = pd.DataFrame({
        'date_prevue': dates_futures,
        'quantite_predite': predictions_brutes
    })

    return serie, df_pred

# Affichage interactif pour un produit donné
def afficher_prediction_produit_interactif2(code_produit, df_top_references_pred, n_semaines_pred=8):
    produit = df_top_references_pred[df_top_references_pred['code'] == code_produit].iloc[0]

    dates_commandees = produit['dates_commandes']
    quantites_commandees = produit['quantites_commandees']
    conditionnement = produit['conditionnement_moyen']

    # Appelle la fonction de prédiction
    serie_hist, df_pred = predire_par_repetition(
        dates_commandees, quantites_commandees,
        n_semaines_pred, conditionnement
    )

    # Prépare les données pour le graphe
    df_hist = serie_hist.reset_index()
    df_hist.columns = ['date', 'quantite']
    df_hist['type'] = 'Historique'
    df_pred['type'] = 'Prédiction'

    # Combine les historiques et les prédictions dans un seul DataFrame
    df_combined = pd.concat([
        df_hist,
        df_pred.rename(columns={'date_prevue':'date', 'quantite_predite':'quantite'})
    ], ignore_index=True)

    # Affiche un graphique interactif
    fig = px.line(df_combined, x='date', y='quantite', color='type',
                  markers=True,
                  title=f"Quantités commandées et prédites – {produit['designation']} ({produit['famille']})",
                  labels={'quantite': 'Quantité', 'date': 'Date'})
    fig.update_layout(xaxis_title='Date (semaine)', yaxis_title='Quantité commandée')
    fig.show()


# #####5.3.1 - Visualisation de la prédiction des commandeds (fréquence)

# In[ ]:


# Appelle la fonction pour un produit donné
afficher_prediction_produit_interactif2('8001118', df_top_references_pred, n_semaines_pred=8)


# In[ ]:


pd.set_option('display.max_rows', None)
display(df_top_references)
pd.reset_option('display.max_rows')


# ###5.4 - Prédiction de la quantité à commander via la gestion du stock/les ventes

# Cette deuxième méthode de prédiction sse base sur un nouveau étalage des ventes qui inclut la simulation d'un stock. Cet étalage se fait en fonction de la fréquence de commande (similaire au fonctionnement de la méthode 1). Pour cela, on initialise le stock de la semaine -1 à 0 (semaine 0 = semaine de la 1ère prédiction). Les scripts analysent les ventes hebdomadaires et déclenche des commandes quand on passe en dessous le seuil de 10% (par rapport au conditionnement).

# In[ ]:


# Une version plus stable et prédictive, utilisable dans des modèles ou des tests reproductibles, l'étalage se fait selon la fréquence
# au lieu d'un nb de semaine fixe, sauf si on manque de données
def etaler_ventes2(ref, date_debut_globale, nb_semaines):
    commandes = [pd.to_datetime(d) for d in ref['dates_commandes']]
    quantites = ref['quantites_commandees']

    ventes = [0] * nb_semaines

    # - de 2 cmd = étalage sur 17sem.
    if len(commandes) < 2:
        date_debut = commandes[0]
        index_debut = (date_debut - date_debut_globale).days // 7
        for j in range(17):
            if 0 <= index_debut + j < nb_semaines:
                ventes[index_debut + j] += quantites[0] / 17
    # Sinon, étalage selon la fréquence de moyenne de cmd
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


# In[ ]:


# Prédit le ca d'un article dont on a renseigné le code
def predire_ca_article(df_ventes_detaillees, code, nb_semaines_a_predire):
    df_article = df_ventes_detaillees[df_ventes_detaillees['code'] == code].copy()
    if df_article.empty:
        raise ValueError("Code produit non trouvé")

    df_article = df_article.sort_values('semaines')
    df_article['ca_cumule'] = df_article['ca_article'].cumsum()
    df_article['num_semaine'] = ((df_article['semaines'] - df_article['semaines'].min()).dt.days // 7)

    model = LinearRegression()
    model.fit(df_article[['num_semaine']], df_article['ca_cumule'])

    last_week = df_article['num_semaine'].max()
    semaines_futures = np.arange(last_week + 1, last_week + 1 + nb_semaines_a_predire)
    pred_cumule = model.predict(semaines_futures.reshape(-1, 1))
    pred_semaine = np.diff(np.concatenate([[df_article['ca_cumule'].iloc[-1]], pred_cumule]))

    return pred_semaine


# In[ ]:


def simuler_stock_et_commandes(qte_initiale, prix, marge, ca_prevu, conditionnement):
    stock = []
    commandes = []
    stock_actuel = qte_initiale

    for ca in ca_prevu:
        qte_sortie = np.ceil(ca / (prix * marge))  # quantités vendues prévues cette semaine

        # 1. Vérifier le stock avant la sortie
        seuil = 0.1 * conditionnement
        if stock_actuel < seuil:
            manquant = max(0, -stock_actuel + seuil)
            n_cond = int(np.ceil(manquant / conditionnement))
            commande = n_cond * conditionnement
        else:
            commande = 0

        # 2. Mettre à jour le stock avec la commande reçue
        stock_actuel += commande

        # 3. Retirer les ventes
        stock_actuel -= qte_sortie

        # 4. Enregistrer
        stock.append(stock_actuel)
        commandes.append(commande)

    return commandes, stock


def calculer_commandes(stock_simule, conditionnement):
    commandes = []
    for s in stock_simule:
        seuil = 0.1 * conditionnement
        if s < seuil:
            manquant = max(0, -s + seuil)
            n_cond = int(np.ceil(manquant / conditionnement))
            commandes.append(n_cond * conditionnement)
        else:
            commandes.append(0)
    return commandes



# In[ ]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def simuler_stock_et_commandes(stock_initial, prix_unitaire, marge, ca_prevu, conditionnement):
    """
    Simule le stock semaine par semaine à partir d'un CA prévisionnel.
    Le stock est reconstitué via des commandes en multiples du conditionnement,
    déclenchées si le stock ne permet pas de couvrir la demande.
    """
    stock = stock_initial
    commandes = []
    stock_semaine = []

    for ca in ca_prevu:
        # Calcule la demande prévue en quantités
        quantite_prevue = ca / (prix_unitaire * marge)

        # Si le stock ne suffit pas, on commande
        if stock < quantite_prevue:
            besoin = quantite_prevue - stock
            commande = int(np.ceil(besoin / conditionnement) * conditionnement)
            stock += commande
        else:
            commande = 0

        # Consomme le stock
        stock -= quantite_prevue
        stock = max(0, stock)  # sécurité pour éviter un stock négatif

        # ✅ Arrondi du stock à l’unité
        stock_arrondi = int(round(stock))

        # Ajoute les résultats de la semaine
        commandes.append(commande)
        stock_semaine.append(stock_arrondi)

        # Le stock pour la semaine suivante est la version arrondie
        stock = stock_arrondi

    return commandes, stock_semaine




# In[ ]:


import plotly.graph_objects as go
import pandas as pd

def afficher_commandes_et_stock(ca_prevu, commandes, stock, start_date=None):
    """
    Affiche un graphique interactif montrant les prévisions de CA, commandes à passer et stock simulé.

    """
    nb_semaines = len(ca_prevu)

    # Si aucune date de départ n'est fournie, on prend le lundi de cette semaine
    if start_date is None:
        today = pd.Timestamp.today()
        start_date = today - pd.Timedelta(days=today.weekday())

    dates = [start_date + pd.Timedelta(weeks=i) for i in range(nb_semaines)]

    # Préparation des courbes
    fig = go.Figure()

    # Courbe 1 : CA prévisionnel (barres)
    fig.add_trace(go.Bar(
        x=dates, y=ca_prevu,
        name="CA prévisionnel",
        marker_color='lightblue',
        yaxis='y2',
        opacity=0.6
    ))

    # Courbe 2 : Commandes à passer (barres)
    fig.add_trace(go.Bar(
        x=dates, y=commandes,
        name="Commandes à passer",
        marker_color='orange',
        opacity=0.7
    ))

    # Courbe 3 : Stock simulé (ligne)
    fig.add_trace(go.Scatter(
        x=dates, y=stock,
        mode='lines+markers',
        name="Stock simulé",
        line=dict(color='green', width=3)
    ))

    fig.update_layout(
        title="Prévision des commandes et du stock",
        xaxis_title="Date (semaine)",
        yaxis=dict(title="Commandes / Stock"),
        yaxis2=dict(title="CA", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        bargap=0.2,
        template="plotly_white"
    )

    fig.show()


# In[ ]:


display(df_synthese)


# In[ ]:


# Test
code = '8001118'
conditionnement = df_synthese[df_synthese['code'] == code]['conditionnement_moyen'].values[0]
prix = df_synthese[df_synthese['code'] == code]['prix_unitaire_moyen'].iloc[0]
marge = 1.3  # Marge moy. de 30%
stock_initial = 0
nb_semaines = 4

# Prédire le chiffre d'affaires futur
ca_pred = predire_ca_article(df_ventes_detaillees, code, nb_semaines)

# Simuler le stock + commandes
commandes, stock = simuler_stock_et_commandes(stock_initial, prix, marge, ca_pred, conditionnement)


afficher_commandes_et_stock(ca_pred, commandes, stock)

print("Commandes à passer sur 4 semaines :", commandes)
print("Stock simulé :", stock)


# #Exportation Git

# In[1]:


get_ipython().system('apt-get install git -y')


# In[ ]:


get_ipython().system('git config --global user.name "owenvary"')
get_ipython().system('git config --global user.email "owen.vary@isen-ouest.yncrea.fr"')


# In[2]:


get_ipython().run_line_magic('cd', '/content')
get_ipython().system('git init')


# In[9]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Colab_Notebooks/AnalyseVentes_SUPERGROUP.ipynb"')


# In[10]:


get_ipython().system('cp "/content/drive/MyDrive/Colab_Notebooks/AnalyseVentes_SUPERGROUP.py" "/content/AnalyseVentes_SUPERGROUP.py"')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




