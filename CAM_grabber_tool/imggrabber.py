#
# imggrabber.py
#
# Date: 18 July 2016 
# Author: Daniel Becker
# Bearbeitet 01.04.20 - Marcus Witzke
# -> 


import sys
#import urllib2

import time
import datetime
import os

import socket
from socket import AF_INET, SOCK_DGRAM


if (sys.version_info > (3, 0)):
     # Python 3 code in this block
     import urllib.request
     print ("Python v3")
else:
     # Python 2 code in this block
     import urllib2
     print ("Python v2")


#
# CONFIGS
#

# http auth
username = 'root'
password = 'ccasct1'

# ip address list cameras
# -> if connected over "Media" use following cam-identifier
# -> commend in if cams available
cameraAdds = [ \
'1r', \
'1c', \
'1l', \
#'2r', \
#'2c', \
#'2l', \

# Examples if script can connect diretly
#'192.168.1.2', \
#'192.168.1.3', \
#'192.168.1.3', \
#'192.168.96.163', \
#'192.168.96.164', \

]

# local image directory
#imgDir = 'imgs'
imgDir = 'imgs/CAM_'



#
# FUNCTIONS
#

def getTimestamp():
	return int(time.time())


def ipAddrToCamId(ipadd):
	# Camera IDs correspond to IP host adress (160 = 0, 161 = 1, ..)
#	hostadd = ipadd.split('.')[3]
#	return (int(hostadd)-160)
	hostadd = ipadd [0]
	return (int(hostadd))

def CamId_ident(ipadd):
	# Camera IDs correspond to IP host adress (160 = 0, 161 = 1, ..)
#	hostadd = ipadd.split('.')[3]
#	return (int(hostadd)-160)
	hostident = ipadd [1]
	return ((hostident))

def saveImgFromURL( url, fname ):

	#Create a password manager
	manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
	#manager.add_password(None, url, username, password)

	#Create an authentication handler using the password manager
	auth = urllib2.HTTPBasicAuthHandler(manager)

	#Create an opener that will replace the default urlopen method on further calls
	opener = urllib2.build_opener(auth)
	urllib2.install_opener(opener)

	#Here you should access the full url you wanted to open
	try:
    		# socketstuff
		response = urllib2.urlopen(url, timeout=5)
		imgdata = response.read()
		f = open(fname,'w')
		f.write(imgdata )
		f.close()
	except socket.timeout:
		print ("Request timeout for ")+url
	except socket.error:
		print ("Socket error for ")+url
	except urllib2.URLError:
		print ("Unknown network exception for ")+url
		


#
# MAIN 
#

print ("Starting..")


# Checks
# -> for every entry in "cameraAdds" check for existing dir
# -> mkdir if dir not exist
for addr in cameraAdds:
	full_imgDir = imgDir+str(ipAddrToCamId(addr))
	if not os.path.exists(full_imgDir):
		os.makedirs(full_imgDir)


while True:
	
	# Start download attempt for all cameras
	#print datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " attempting to download images from " + str(len(cameraAdds)) + (" cameras..")
	print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + str(len(cameraAdds)))

	# Iterate through all cameras and fetch images
	for addr in cameraAdds:
	#    imgurl = "http://"+addr+"/axis-cgi/jpg/image.cgi"
		imgurl = "https://media.dcaiti.tu-berlin.de/tccams/"+addr+"/jpg/image.jpg??camera=1&resolution=1280x720&rotation=0&audio=0&mirror=0&fps=0&compression=50"
		imgfile = imgDir+str(ipAddrToCamId(addr))+"/CAM_"+str(ipAddrToCamId(addr))+str(CamId_ident(addr))+"_"+str(getTimestamp())+".jpg"
		saveImgFromURL(imgurl, imgfile)
	    
	# Repeat every 30 minutes
	time.sleep(0.5)













