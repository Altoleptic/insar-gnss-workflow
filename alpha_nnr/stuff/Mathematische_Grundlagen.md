# Mathematische Grundlagen

## Least Squares Methode
Die Methode der kleinsten Quadrate (Least Squares) ist ein Verfahren, um die besten Parameter für ein Modell zu finden, indem die Abweichungen zwischen den beobachteten und den modellierten Werten minimiert werden. Dabei werden die Abweichungen quadriert, um negative Werte positiv zu machen und größere Abweichungen stärker zu gewichten.

- **Mathematische Darstellung**:
  Für eine Menge von Beobachtungen y und Modellwerten f(x) wird die Summe der quadrierten Abweichungen minimiert:
  Minimiere Summe((y - f(x))^2)

## Haversine-Distanz
Die Haversine-Distanz berechnet die kürzeste Entfernung zwischen zwei Punkten auf einer Kugeloberfläche, basierend auf ihren Breitengraden und Längengraden.

- **Formel**:
  d = 2 * R * arcsin(sqrt(sin^2(Delta_phi / 2) + cos(phi_1) * cos(phi_2) * sin^2(Delta_lambda / 2)))

  Dabei bedeuten:
  - d: die berechnete Entfernung zwischen den Punkten.
  - R: der Radius der Kugel (z. B. der Erdradius).
  - Delta_phi: die Differenz der Breitengrade.
  - Delta_lambda: die Differenz der Längengrade.
  - phi_1, phi_2: die Breitengrade der beiden Punkte.

## LOS-Projektion
Die Projektion in die Sichtlinie (Line-of-Sight, LOS) berechnet die Bewegung entlang einer bestimmten Richtung.

- **Mathematische Darstellung**:
  LOS-Verschiebung = v_x * l_x + v_y * l_y + v_z * l_z

  Dabei bedeuten:
  - v_x, v_y, v_z: die Komponenten des Verschiebungsvektors.
  - l_x, l_y, l_z: die Komponenten des Richtungsvektors.

## Amplitude-Berechnung
Die Amplitude beschreibt die maximale Schwankung einer Größe über einen Zeitraum. Sie wird als die Hälfte des Unterschieds zwischen dem maximalen und minimalen Wert definiert.

- **Formel**:
  Amplitude = (max(Wert) - min(Wert)) / 2

## Lineare Regression
Die lineare Regression ist ein Verfahren, um eine Linie zu finden, die die Beziehung zwischen zwei Variablen beschreibt. Sie verwendet die Methode der kleinsten Quadrate, um die Abweichungen zwischen den beobachteten und den modellierten Werten zu minimieren.

- **Mathematische Darstellung**:
  y = m * x + b

  Dabei bedeuten:
  - y: die abhängige Variable.
  - x: die unabhängige Variable.
  - m: die Steigung der Linie.
  - b: der Achsenabschnitt der Linie.

---

Diese mathematischen Methoden sind allgemeingültig und bilden die Grundlage für viele Anwendungen in der Geodäsie und Datenanalyse.
