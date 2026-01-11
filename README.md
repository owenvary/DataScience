## Structure du projet

```text
analyse-ventes-app/
│
├── Data/
│   ├── Fichiers CSV/ 
│     ├── df_traitee.csv/        # Df traitée contenant l'ensemble des données de vente des 5 fournisseurs
│     ├── nt_apdv.csv/           # Df non traitée du fournisseur "Au pied de la vigne"
│     ├── nt_autentik.csv/       # Df non traitée du fournisseur "L'autentik"
│     ├── nt_cheritel.csv/       # Df non traitée du fournisseur "Cheritel"
│     ├── nt_lbb.csv/            # Df non traitée du fournisseur "Les bonnes bouteilles"
│     └── nt_supergroup.csv/     # Df non traitée du fournisseur "Supergroup"
│
├── scripts/               
│   ├── oauth_init.py/           # Script d'initalisation des tokens de l'API OAuth 
│   └── .ouath_tokens/           # Dossier contenant l'ensemble des tokens actifs générés 
│     └── token_xxx@xxx.com.json # Exemple de token 
│ 
├── src/               
│   ├── gestion_factures.py/     # Script de gestion des factures (téléchargement, extraction, rangement, renommage, ...)
│   ├── google_drive_manager.py/ # Script de gestion de la connexion au répertoire Google drive (récupère l'id dossier, récupère l'ensemble des factures présent dans le répertoire) 
│   ├── oauth_drive_manager.py/  # Script de gestion des tokens (renouvellement des tokens si la periode de validité a expiré) 
│   ├── traiter_factures.py/     # Script de traitement des factures des 5 fournisseurs 
│   └── visualiser.py/           # Script de visualisation qui permet de visualiser l'ensemble des données (df, données filtrées, graphiques, indicateurs, ...) 
├── app.py/                      # Equivalent d'un main
├── streamlit_app.py/            # Script de la structure complète de l'application streamlit
└── Guide d'utilisation App_analyse_ventes.pdf # Guide d'utilisation pdf
## Structure du projet
```
