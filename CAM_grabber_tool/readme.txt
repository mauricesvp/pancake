#
# imggrabber_version.py
#
# Date: 19 January 2021 
# Author: Marcus Witzke

Sriptname: 		Imggrabber_tool
Verwendung: 	Das Skript nimmt von einer Liste von Netzwerkkameras Einzelbilder mit 2Hz auf und speichert 
				die Bilder unter Angabe des Timestamps je Kamera in separaten Ordnern. Das Skript hatte 
				Daniel mal für die Kamera Kalibrierung geschrieben. Habe es etwas abgewandelt.
 

Aufruf:			Aufruf mittels Python (derzeit v3) -> „python3 CAM_grabber_tool/imggrabber.py"
 
Ordnerstruktur: Das Skript sucht im "Skriptverzeichnis" nach einem Unterordner "/img". In diesem 
				Verzeichnis werden dann automatisch Unterordner für alle Kameras angelegt. Die 
				Ordnerstruktur müsste die folgende sein:

				-> CAM_grabber_tool (mit dem imggrabber.py-Script)
				-> CAM_grabber_tool/imgs/… (in diesem Ordner werden die Unterordner angelegt, die als Input-IP (hier „1r“ & „2r“) angegeben wurden)
 
				Ausgegebene Datei wird dann wie folgt abgelegt „CAM_1r_1585745435.jpg“ oder 
				„CAM_2r_1585745436.jpg“