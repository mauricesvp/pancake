# Evaluation

## Evaluation Run [August 6th, 2021]
The sample dataset used was [r44](https://drive.google.com/drive/folders/1p65faGMFBUeIgWYBBQYAkjXfAGQVpBk7?usp=sharing). For the evaluation we only focused on the upper lane (right side from Ernst-Reuter-Platz to Charlottenburger Tor).

1. Phase: Tracker Evaluation ([DeepSORT](../../docs/modules/tracker.md#DeepSORT), [Centroid Tracker](../../docs/modules/tracker.md#Centroid))
2. Phase: Detector Evaluation ([YOLOv5](../../docs/modules/detector.md): pre-trained, custom-trained)
3. Phase: Backend Evaluation ([SIMPLE](../../docs/modules/backends.md#Basic), [DEI(simple), DEI(normal)](../../docs/modules/backends.md#DEI))

### A,B,C-Zones

<img src="/gitimg/eval_060821_evalzones.png">

### Results
#### Phase 1
First of all, we compare the different trackers.

<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
  <td><pre><b>DeepSORT</b>
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
<td></td>
</tr>
<tr>
  <td><pre><b>Centroid Tracker</b>
YoloV5m(pre)
DEI(Normal)</pre></td>
<td>3</td>
<td>8 von 8</td>
<td>6 von 6</td>
<td>3 von 3</td>
<td>3 von 3</td>
<td></td>
</tr>
</tbody>
</table>


#### Phase 2
Next, the detectors are compared.

<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><pre>DeepSORT
<b>YoloV5m(pre)</b>
DEI(Normal)</pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
<td>(Same as Phase 1 Run 1)</td>
</tr>
<tr>
<td><pre>DeepSORT
<b>YoloV5m(custom)</b>
DEI(Normal)</pre></td>
<td>2</td>
<td>5 von 8</td>
<td>2 von 6</td>
<td>1 von 3</td>
<td>0 von 3</td>
<td></td>
</tr>
</tbody>
</table>


#### Phase 3
Last but not least, the different backends are compared.

Using DeepSORT:

<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><pre>DeepSORT
YoloV5m(pre)
<b>Simple</b></pre></td>
<td>7</td>
<td>2 von 8</td>
<td>2 von 6</td>
<td>0 von 3</td>
<td>0 von 3</td>
<td></td>
</tr>
<tr>
<td><pre>DeepSORT
YoloV5m(pre)
<b>DEI(Simple)</b></pre></td>
<td>2,5</td>
<td>4 von 8</td>
<td>4 von 6</td>
<td>2 von 3</td>
<td>2 von 3</td>
<td></td>
</tr>
<tr>
<td><pre>DeepSORT
YoloV5m(pre)
<b>DEI(Normal)</b></pre></td>
<td>2</td>
<td>6 von 8</td>
<td>4 von 6</td>
<td>3 von 3</td>
<td>2 von 3</td>
<td>(Same as Phase 1 Run 1)</td>
</tr>
</tbody>
</table>

Using Centroid Tracker:

<table>
<thead>
<tr>
<th>Configurations</th>
<th>Avg. FPS</th>
<th>A</th>
<th>B</th>
<th>C</th>
<th>Frame</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><pre>Centroid Tracker
YoloV5m(pre)
<b>Simple</b></pre></td>
<td>2</td>
<td>4 von 8</td>
<td>4 von 6</td>
<td>0 von 3</td>
<td>0 von 3</td>
<td></td>
</tr>
<tr>
<td><pre>Centroid Tracker
YoloV5m(pre)
<b>DEI(Simple)</b></pre></td>
<td>4</td>
<td>6 von 8</td>
<td>5 von 6</td>
<td>3 von 3</td>
<td>1 von 3</td>
<td></td>
</tr>
<tr>
<td><pre>Centroid Tracker
YoloV5m(pre)
<b>DEI(Normal)</b></pre></td>
<td>3</td>
<td>8 von 8</td>
<td>6 von 6</td>
<td>3 von 3</td>
<td>3 von 3</td>
<td></td>
</tr>
</tbody>
</table>

### Summary
Das Custom Training verschlechtert die Ergebnisse, da insb. nur auf Daten von Autos auf der Mittelkamera zum Training verwendet wurden.

Generell besseres Tracking auf der oberen Fahrbahn, da die Bäume die Sicht auf die untere Fahrbahn teilweise verdecken.
