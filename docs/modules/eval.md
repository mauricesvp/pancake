# Evaluation

## Evaluation Run [August 6th, 2021]
The sample dataset used was [r44](https://drive.google.com/drive/folders/1p65faGMFBUeIgWYBBQYAkjXfAGQVpBk7?usp=sharing). For the evaluation we only focused on the upper lane (right side from Ernst-Reuter-Platz to Charlottenburger Tor).

1. Phase: Tracker Evaluation ([DeepSORT](../../docs/modules/tracker.md#DeepSORT), [Centroid Tracker](../../docs/modules/tracker.md#Centroid))
2. Phase: Detector Evaluation ([YOLOv5](../../docs/modules/detector.md): pre-trained, custom-trained)
3. Phase: Backend Evaluation ([SIMPLE](../../docs/modules/backends.md#Basic), [DEI(simple), DEI(normal)](../../docs/modules/backends.md#DEI))

[Jump to Remarks](#summary)

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
<td>9</td>
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
<td>(Same as Phase 1 Run 2)</td>
</tr>
</tbody>
</table>

### Summary

#### General Remarks
Tracking is generally better on the upper lane, as a lot of trees cover the lower lane, which is why we limited our evaluation to the upper lane.

We also choose a dataset which provided fairly favorable results, i.e. the datasets has only few cars, no shadows, and normal lighting.
Given heavy traffic, large shadows, and/or bad lighting conditions, it must be assumed that the results can be (a lot) worse.

Our evaluation shows that the configuration which offers the best tradeoff between speed and accuracy is using the <b>pretrained YOLOv5 Detector</b>, <b>DEI(normal) Backend</b> and <b>Centroid Tracker</b>.

#### Phase 1
The Centroid Tracker not only runs faster, it also produces very accurate results. This is due to the detections being sufficiently consistent, as well as optimizations targeted specifically towards our use case.

DeepSORT only produces mixed results, which is mostly caused by the feature extractor which was not trained for our use case.

#### Phase 2
Overall, the pretrained Detector produces better results than the custom trained Detector. This is a result of the training data, which only included cars from the middle camera perspective. Ideally, the YoloV5 Detector would have been trained with a lot of data from all camera angles, which should then outperform the pretrained Detector.

The results by the pretrained Detectorr however are already very decent.

#### Phase 3
Generally speaking, the more simpler the backend, the more fps, but also less accuracy is achieved.

For our use case the Simple Backend is not a feasible option, however given improvements on the Detector, the DEI Backend could be simplified further, which would result in less of a slowdown.
