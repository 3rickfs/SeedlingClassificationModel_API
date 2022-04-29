from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.shortcuts import render
from . IA_ModelsToolBox import get_seedling_class
from django.conf import settings
import pandas as pd
import os
import requests
from PIL import Image

# Create your views here.
#@api_view(['GET'])
#def index_page(request):
#	return_data	= {
#		"error": "0",
#		"message": "Successful",
#	}
#	return Response(return_data)

@api_view(['POST'])
def predict_seedlingClass(request):
	try:
		#prediction process
		print("Loading image file")
		url = request.POST.get('url') #form-data (key-value) format to store the request
		print(f"The url is: {url}")
		head, tail = os.path.split(url)
		r = requests.get(url, allow_redirects=True)
		imgfolder_path = settings.MEDIA_ROOT
		open(os.path.join(imgfolder_path, tail), 'wb').write(r.content)
		print("Image loaded")

		files = []

		for i in os.listdir(imgfolder_path):
			print("Reading an image file")
			img = Image.open(os.path.join(imgfolder_path,i))
			fmt = str(img.format)
			#if i.endswith('.jpg') or i.endswith('.png'):
			if fmt == "JPEG" or fmt == "JPG" or fmt == "PNG":
				files.append(i)
			else:
				print(f"{i} has not a valid format")

		preds = []
		for j in files:
			print("Performing model processing")
			imgpath = os.path.join(imgfolder_path, j)
			preds.append(get_seedling_class(settings.MODELS, imgpath))

		print("Model has got the predictions")
		print(f"Number of predictions done {len(preds[0])}") #First considering one image, if there is more will display the last one image preds
		preds_df = pd.DataFrame(data=preds[0], columns=['class','x','y','w','h','confidence'])
		print(preds_df)
		preds_jso = preds_df.to_json(orient='values')
		#Removing the downloaded file to avoid innecesary processing in the next pred
		os.remove(imgpath)
		
		return JsonResponse(preds_jso, safe=False)

	except Exception as e:
		predictions = {
			'error': '2',
			"message": str(e),
		}		

	return JsonResponse(predictions, safe=False)


