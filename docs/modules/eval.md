# Evaluation

## Evaluation Run [August 6th, 2021]
Testdata used is the r44 sample set.

1. Phase: Tracker Evaluation ([DeepSORT](../../docs/modules/tracker.md#DeepSORT), [Centroid Tracker](../../docs/modules/tracker.md#Centroid))
2. Phase: Detector Evaluation ([YOLOv5](../../docs/modules/detector.md): pre-trained, custom-trained)
3. Phase: Backend Evaluation ([SIMPLE](../../docs/modules/backends.md#Basic), [DEI(simple), DEI(normal)](../../docs/modules/backends.md#DEI))

### A,B,C-Zones

<img src="/gitimg/eval_060821_evalzones.png">

### Results
#### Phase 1
<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
</tr>
</thead>
<tbody>
<tr>
<td><pre>DeepSORT
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
</tr>
<tr>
<td><pre>Centroid Tracker
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
</tr>
</tbody>
</table>
<!---<td><pre>Die meisten Autos werden über weite Teile der Strecke erfolgreich getrackt.
Einige Autos verlieren über die gesamte Breite 1-2 mal die Tracking ID, einige auch häufiger.
Nach Zuordnung neuer ID werden diese jedoch solide weitergetrackt.
Generell besseres Tracking auf der oberen Fahrbahn, insb. da die Bäume die Sicht auf die untere Fahrbahn verdecken.
</pre></td>--->


#### Phase 2
<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
</tr>
</thead>
<tbody>
<tr>
<td><pre>DeepSORT
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
</tr>
<tr>
<td><pre>Centroid Tracker
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
</tr>
</tbody>
</table>


#### Phase 3
<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
</tr>
</thead>
<tbody>
<tr>
<td><pre>DeepSORT
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
</tr>
<tr>
<td><pre>Centroid Tracker
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
</tr>
</tbody>
</table>

### Summary
Foo
