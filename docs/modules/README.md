# Modules

<img width="800" height="%" src="/gitimg/app_structure.png">

# Backend

| Backend       | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| Basic         | foo       | ```"simple"```
| DEI           | foo       | ```"dei"```

Because the Detection can - depending on the data - not necessarily be run directly,
the Backend is responsible for adjusting the data as necessary to make sure the results are in order.
All backends are initialized with an instance of a Detector, which is used for the detection.
<br>
<details>
  <summary><b>Basic</b></summary>
  The Basic Backend simply takes the input image(s), and runs the detection on each image.
</details>
<details>
  <summary><b>DEI (Divide and Conquer)</b></summary>
  The DEI Backend is specifically designed for the detection on the Strasse des 17. Juni,
  using a panorama image (made up by three images).
  Because the detections would be very poor if it was run one the panorama directly,
  the Backend first splits the panorama image into partial images.
  These then get rotated, depending on the proximity to the center (no rotation in the center, more rotation on the outer sides).
  This is done as the angle of the cars gets quite skewed on the outer sides, which hinders a successful detection.
  The actual detection is now run on the partial images, after which the rotation und splitting are reversed to produce the final results.
</details>
<details>
  <summary><b>Adding a new Backend</b></summary>
  <ol>
    <li>Create your backend_foo.py within <code>detector/backends/</code> .</li>
    <li>Create a Backend class that inherits from the <a href="pancake/detector/backends/backend.py">Base Backend</a>.</li>
    <li>Implement the <code>detect</code> method.</li>
    <li>Add your Backend to the <a href="pancake/detector/backends/__init__.py">registry</a> (i.e. add <code>from .backend_foo import Foo</code>).</li>
    <li>Set your Backend in the configuration (under "BACKEND" -> NAME: "foo").</li>
  </ol>
Important: When implementing your Backend, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/detector/backends/backend.html> Backend API</a>!
</details>
