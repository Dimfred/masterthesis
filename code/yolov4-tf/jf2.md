# Ideas

- Klassen als ensemble verwenden?
- alle offline augmentierungen derotieren => Mehrheitsentscheid

- Rotations layer?

# Grid detection

## Idee 1

- Grid und Zeichnung sind sich sehr ähnlich
- Finde Zeichnungs Farbe indem man sich dominante Farben in Bounding Box anschaut?
- BUT Zeichnung hat gleiche Farbe => fucked

- Canny edge findet grid und Zeichnung
- Canny Maske nutzen um richtige colors aus dem Image zu extrahieren
- Thresholden mit gefundenen Farben

## Idee 2

- wieder Farben in Bboxen anschauen
- 3 Cluster verwenden? (white, grid, circuit)
- dynamisch Canny threshold ermitteln Grid fällt dann durch

## Idee 2.1

- gradient innerhalb von boundingboxen berechnen => daraus canny th ableiten
- RESULTS:
    - funktioniert in 50% der Fälle
    - abhängig von der stärke des blurs
        - vll blur größe finden basierend auf ???

## Idee 3

- template matching (somehow)
- mit ORB (findet jedoch keine Punkte im Grid)
-

## Idee 4

- Hough lines finden
- lines = [*perpendicular_lines, *parrallel_lines]
- PROBLEME: rauschen von nicht gecropten Bildern
- guten Winkelthreshold für Parallelität

## Idee 5

- unet biatch

## Idee 6

- fit squares, rebuild the pattern, and exclude it

## Idee 7

- fit a template pattern with regression?
