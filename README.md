# Aktien-Vorhersage mit historischen Daten als Training
### Was ist das hier überhaupt?

Mit diesem Code wird probiert anhand von Aktiendaten des heutigen Tages zu prognostizieren, ob morgen der Wert der Aktie steigen wird. Dazu wird die Bibliothek **sckit-learn** für die benötigte Lernlogik verwendet. Der ganze Code ist in Python geschrieben


### Wie installiere ich denn überhaupt Python?
Zuerst einmal überprüfe bitte ob Python an deinem Betriebssystem installiert ist. Sonst besuche bitte [die Website von Python](https://www.python.org/downloads/) und lade dir am aller besten Python3.11 herunter.

### Wie lade ich mir dann das Projekt herunter?
Anschließend lade dir das Projekt über den grünen Button `Code -> Download ZIP` herunter oder klone es über das Terminal in das richtige Verzeichnis.
Um es über das Terminal zu klonen führe diesen Code nacheinander aus
```
cd /Pfad/zu/deinem/Verzeichnis
git clone https://github.com/einsaetze-finanzen/stockPrediction.git
```

Jetzt müsste in deinen Dateien im Ordner stockPrediction alle Dateien des Projekts zu finden sein

### Wie installiere ich die benötigten Frameworks?
Wie du vielleicht schon gesehen hast, gibt es in den Dateien eine Datei genannt [requirements.txt](https://github.com/einsaetze-finanzen/stockPrediction/blob/main/requirements.txt).
Sie enthält alle Frameworks und ihre Versionen, die für dieses Projekt benötigt werden.
Um sie alle zu installieren, erstelle am besten erst einmal eine virtuelle Umgebung, am **Mac** z.B mit:
```
python3 -m venv .venv
```
Und anschließend, um die Umgebung zu aktivieren:
```
source .venv/bin/acitvate
```

Installiere daraufhin alle Abhängigkeiten, mithilfe von (wieder Mac):
```
pip3 install -r requirements.txt
```

Um zu überprüfen, ob alle Frameworks richtig installiert wurden, kannst du `pip3 freeze` ausführen.

**Wichtig:** Führe alle diese Codestücke im Verzeichnis von stockPrediction aus, also z.B `/Pfad/zu/deinem/Verzeichnis/stockPrediction`. Den Pfad kannst du mithilfe von [`cd`](https://phlow.de/magazin/terminal/datei-ordner-befehle/) im Terminal ändern

### Einfachere Methode
Eine andere und einfachere Möglichkeit ist dir einen Code-Editor wie z.B. [VSCode von Microsoft](https://code.visualstudio.com/download) zu installieren. Er hat auch Funktionen, wie das Erstellen einer virtuellen Umgebung und installieren von requirements.txt. Öffne natürlich zuerst den Ordner stockPrediction in VSCode und evtl. musst du auch Erweiterungen in der Seitenleiste, wie z.B. Python installieren.

### Ausführen

Erstelle eine Datei mit dem Namen `.env`. Schreibe in diese Datei:
```
PATH = "Pfad/zu/Datei"
```
Ersetze `Pfad zu Datei` hier duch den Pfad zu **predModel.py**, also z.B. `C:/Users/XXX/Documents/stockPreditction`.

Führe den Code entweder im Terminal durch `python3 predModel.py` aus oder in VSCode mithilfe von der Pfeil-Taste oben rechts, wenn du **main.py geöffnet** hast. Evtl. musst du zuvor in der Leiste rechts unten auf Python3.11 drücken und zu .venv ändern

### Anpassen

Wenn der Code richtig funktioniert, kann man im Terminal die Genauigkeit des Codes sehen. Man kann mithilfe von Ai-Assistenten, wie z.B. ChatGPT oder Blackbox.ai (für Coding besser), probieren den Code zu verstehen und weiterzuentwickeln


