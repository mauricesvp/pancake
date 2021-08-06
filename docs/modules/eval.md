# Evaluation

## Evaluation Run [August 6th, 2021]
Testdata used is the r44 sample set.

1. Phase: Tracker Evaluation ([DeepSORT](../../docs/modules/tracker.md#DeepSORT), [Centroid Tracker](../../docs/modules/tracker.md#Centroid))
2. Phase: Detector Evaluation ([YOLOv5](../../docs/modules/detector.md): pre-trained, custom-trained)
3. Phase: Backend Evaluation ([SIMPLE](../../docs/modules/backends.md#Basic), [DEI(simple), DEI(normal)](../../docs/modules/backends.md#DEI))


--- A B C zone picture ---
<img src="/gitimg/eval_060821_evalzones.png">
A, B, C

| Configurations   | Avg. FPS   |  A  |   B  |  	C	  |  Frame	|  Summary  |
| -------------    | -------    | ----|------|--------|-----    |-----------|
|  DeepSort YoloV5m (pre) DEI (Normal)  | 2           |   6 von 8  |   4 von 6   |   3 von 3     |   2 von 3      |    Die meisten Autos werden über weite Teile der Strecke erfolgreich getrackt. Einige Autos verlieren über die gesamte Breite 1-2 mal die Tracking ID, einige auch häufiger. Nach Zuordnung neuer ID werden diese jedoch solide weitergetrackt. Generell besseres Tracking auf der oberen Fahrbahn, insb. da die Bäume die Sicht auf die untere Fahrbahn verdecken. |
