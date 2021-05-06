"""
imggrabber.py

Date: 		18 July 2016 
Author: 	Daniel Becker
Bearbeitet: 14.01.21 - Marcus Witzke
      	    29.04.21 - Tuan Anh Roman Le
"""
import sys
import time
import datetime
import os
from collections import deque

import socket
from socket import AF_INET, SOCK_DGRAM
import _thread
import concurrent.futures

py_version = sys.version_info
assert (py_version > (3, 0)
 ), f'Python version 3.x.x required (your version: {py_version[0]}.{py_version[1]}.{py_version[2]})'
import urllib.request



""" CONFIGS """
# url
URL_PREFIX = "https://media.dcaiti.tu-berlin.de/tccams/"
URL_SUFFIX = "/jpg/image.jpg??camera=1&resolution=1280x720&rotation=0&audio=0&mirror=0&fps=0&compression=50"

# http auth for cam access
username = 'root'
password = 'ccasct1'

# Cam idents
CAM_IDS = ['1']
IDENTS = ['l', 'c', 'r']

# Thread worker
img_buffer_size = 20

# ----------------------- not used ------------------------------
# ip address list cameras
# -> if connected over "Media" use following cam-identifier
cameraAdds = [ \
'1l', \
'1c', \
'1r', \
#'2r', \
#'2c', \
#'2l', \
]

# Examples if script can connect diretly
#'192.168.1.2', \
#'192.168.1.3', \
# ----------------------------------------------------------------

# local image directory
imgDir = 'imgs/CAM_'

def connectHttp():
	#Create a password manager
	manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
	#manager.add_password(None, url, username, password)

	#Create an authentication handler using the password manager
	auth = urllib.request.HTTPBasicAuthHandler(manager)

	#Create an opener that will replace the default urlopen method on further calls
	opener = urllib.request.build_opener(auth)
	urllib.request.install_opener(opener)

def getTimestamp():
	"""
	returns current time
	"""

	return int(time.time())

def ipAddrToCamId(ipadd: str):
	"""
	returns Cam ID
	"""

	# Camera IDs correspond to IP host adress (160 = 0, 161 = 1, ..)
	#	hostadd = ipadd.split('.')[3]
	#	return (int(hostadd)-160)

	hostadd = ipadd[0]
	return (int(hostadd))

def CamId_ident(ipadd):
	"""
	returns perspective ('l', 'c', 'r')
	"""

	# Camera IDs correspond to IP host adress (160 = 0, 161 = 1, ..)
	#	hostadd = ipadd.split('.')[3]
	#	return (int(hostadd)-160)

	hostident = ipadd[1]
	return ((hostident))

def saveImgFromURL(url, fname):
	#Here you should access the full url you wanted to open
	try:
    	# socketstuff
		response = urllib.request.urlopen(url, timeout=5)
		imgdata = response.read()
		f = open(fname,'wb')
		f.write(imgdata)
		f.close()
	except socket.timeout:
		print ("Request timeout for ")+url
	except socket.error:
		print ("Socket error for ")+url
	except urllib.error.URLError:
		print ("Unknown network exception for ")+url

class CAM():
	def __init__(self, id, img_dir, idents: list):
		self.id = id
		self._target_dir = img_dir+id
		self._idents = idents

		self.addr = [f'{self.id}{ident}' for ident in self._idents]

		self.create_dirs()

	def create_dirs(self):
		for ident in self._idents:
			path = self._target_dir + f'/{ident}'
			if not os.path.exists(path):
				os.makedirs(path)

if __name__ == "__main__":
	connectHttp()
	print ("Starting..")

	CAMS = {f'CAM_{id}': CAM(id=id, img_dir=imgDir, idents=IDENTS) 
		for id in CAM_IDS}

	l_queue = deque()
	c_queue = deque()
	r_queue = deque()

	# frames = 0
	while True:
		# Start download attempt for all cameras
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

		# Iterate through all cameras and fetch images
		# change 'imgurl' in follows if using skript with direct ip request
		CAM = CAMS['CAM_1']

		for addr in CAM.addr:
			# imgurl = "http://"+addr+"/axis-cgi/jpg/image.cgi" // for use when in same network (able to access cam directly)
			img_url = URL_PREFIX+addr+URL_SUFFIX

			img_file = (
				CAM._target_dir							# 'imgs/CAM_1'
				+ f'/{addr[1]}/'						# '/l/'
				+str(getTimestamp())+'.jpg')			# '1619716857'
			
			_thread.start_new_thread(
				saveImgFromURL, (img_url, img_file))
		
		# img_urls = [URL_PREFIX+addr+URL_SUFFIX for addr in CAM.addr]

		# with concurrent.futures.ThreadPoolExecutor() as executor:
		# 	futures = [executor.submit(saveImgFromURL, addr) 
		# 		for addr in CAM.addr]
			
		# 	for future in concurrent.futures.as_completed(futures):
		# 		try:
		# 			data, addr = future.result()
		# 		except Exception as exc:
		# 			raise Exception(f"{exc}")
		# 		else:
		# 			if addr[1] == 'l':
		# 				l_queue.extend(data)
		# 			elif addr[1] == 'c':
		# 				c_queue.extend(data)
		# 			elif addr[1] == 'r':
		# 				r_queue.extend(data)
		
		# Repeat every x seconds
		time.sleep(0.1)













