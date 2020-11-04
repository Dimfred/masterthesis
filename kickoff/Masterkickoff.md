# Masterkickoff

## Classes

- diode (rotation is crucial here)
- capacitor
- resistor (de / us)
- inductor (de / us)
- lamp (de)
- ground
- battery (small / big line)
- transistor?

Q: Classes are with orientation encoded. Maybe remove?

Q: Which classes to add / remove?

- only de or us?
- add connections cross / skips?

## Constraints

1. Only vertical / horizontal OR try to find orientation
2. No skip? No cross?
3. Only white paper?

## Preprocessing

- blur
- binarization / (maybe just greyscale)
- close / open
- thinning?

## Augmentation

- each class rotated 3x 90°.
- further improvements:
  - distortions inside the BoundingBox (BB)

## Done

- trained tinyyolov2 with darknet

## Missing

1. Trained yolov4 in pytorch
2. Creating a "syntax tree" for the circuit
3. Converting the "syntax tree" to ltspice

## Next steps

1. Make clean and train yolo in pytorch
2. Convert detected circuit in "syntax tree"
1. TODO research
   1. Remove detected components
   2. Find wires
   3. Find connectons
4. Convert "syntax tree" into ltspice

## Next optional

4. OCR!
5. Improve recognition by applying reasoning with near characters?

## Evaluation Metrics to use

?

## Thesis questions

1. How much pages?
2. Template?
3. How deep should the theory be? (e.g. explain yolo indepth?)
4. German / English?
6. Where can I train?
