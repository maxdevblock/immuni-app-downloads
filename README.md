# immuni-app-downloads

Backend in `python` per stima downloads dell'app Immuni tramite inferenza Bayesiana.

# Requisiti

- `Python` v3.7
- Consigliato `conda` environment
- `python -m pip install -r requirements.txt`

# Script

Lo script _immuni.py_ ogni mattina alle 7:00 aggiorna il numero di recensioni ricevute su Play Store e App Store e calcola la stima dei downloads.

! Commentare la riga con `copyfile(...)` (utilizzata per aggiornamento del sito https://maxpierini.it/ncov)
