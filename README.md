# ASL Web App

Aplicatie web pentru recunoasterea si generarea limbajului semnelor americane (ASL).

## Doua moduri

1. **Recunoastere semne** (Tab 1)
   - Camera live -> MediaPipe Hands -> KNN pe semnele invatate
   - Recunoaste DOAR semnele predate explicit de utilizator (mod adaptiv)
   - Mod de predare: 2s countdown + 3s capturare a unui set de exemple
   - Blacklist: dupa 5 stergeri ale aceluiasi cuvant, este ignorat permanent
   - Cuvintele detectate -> propozitie naturala

2. **Text -> Semne** (Tab 2)
   - Scrii o propozitie in engleza
   - Aplicatia gaseste videoclipuri pentru fiecare cuvant in dictionarul WLASL (2000 semne)
   - Concateneaza clipurile intr-un singur videoclip ASL pe care il afiseaza in player
   - Cuvinte filler (a, the, is, are, ...) sunt ignorate automat
   - Stemming simplu: "books" -> cauta si "book", "loving" -> cauta si "love"

## Structura

```
ASL_Web_App/
├── server.py              FastAPI + WebSocket (entry point)
├── config.py              Cai catre date + tunabile
├── text_to_sign.py        Text -> video ASL
├── sentence_builder.py    Cuvinte -> propozitie naturala
├── recognizers/
│   └── smart.py           Recunoastere KNN + MediaPipe
├── static/
│   ├── index.html         UI cu cele 2 tab-uri
│   ├── css/style.css
│   ├── js/app.js
│   └── generated_signs/   Cache video-uri generate (creat automat)
├── data/
│   ├── learned_signs.pkl  Semnele invatate (creat automat)
│   └── blacklist.json     Cuvinte interzise (creat automat)
└── requirements.txt
```

## Date externe

Aplicatia foloseste corpus-ul WLASL (2000 semne, ~12.000 videoclipuri).
Acestea NU sunt incluse in proiect (ar fi ~10 GB). Caile sunt configurate
in `config.py`:

```python
WLASL_ROOT = Path(r"C:\Users\Alina\Desktop\claude\ASL Project")
WLASL_JSON = WLASL_ROOT / "WLASL_v0.3.json"
WLASL_VIDEOS_DIR = WLASL_ROOT / "videos"
```


## Rulare

```
cd ASL_Web_App
pip install -r requirements.txt
python server.py
```

Apoi deschide `http://localhost:8000` in browser.

## Detalii tehnice

### Recunoastere (Tab 1)

- **MediaPipe Hands** detecteaza 1-2 maini, returneaza 21 landmark-uri / mana.
- Fiecare mana este normalizata: tradusa la wrist origin si scalata la 1.0
  (invariant la pozitia in cadru si la distanta fata de camera).
- Vector caracteristic: `2 maini * 21 puncte * 3 coordonate = 126 dimensiuni`.
- **Clasificare KNN cu distanta cosine** (k=5), cu prag maxim 0.45 si
  minim 3 exemple per semn. Anti-flicker: smoothing pe ultimele 3 cadre.
- **De ce nu folosim un model preantrenat:** modelele standard (MSASL,
  Kaggle) sunt antrenate pe semnatari diferiti, in conditii diferite
  de iluminat/camera/unghi. Performanta lor pe un utilizator nou este
  slaba (zgomot 2-5%). KNN adaptiv pe exemple personale ofera precizie
  mult mai buna pe semnele invatate de utilizator.

### Generare (Tab 2)

- WLASL JSON mapeaza fiecare cuvant (gloss) la o lista de instante video.
- Pentru fiecare cuvant din propozitie, alegem prima instanta locala disponibila.
- **Concatenare cu ffmpeg `concat` filter** (nu demuxer): fiecare clip este
  scalat la 480x480, padding negru, framerate 25 fps, codec h264. Apoi
  toate clipurile normalizate sunt concatenate. Aceasta abordare functioneaza
  corect chiar si cand sursele au rezolutii sau framerate-uri diferite.
- ffmpeg vine bundle-uit prin `imageio-ffmpeg` (nu trebuie sa-l instalezi
  separat).
- Rezultatele sunt cache-uite dupa hash-ul listei de clipuri: aceeasi
  propozitie generata a doua oara este servita instant.
