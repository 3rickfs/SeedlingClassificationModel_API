from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from . IA_ModelsToolBox import 

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
		#...
	except Exception as e:
	predictions = {
		'error': '2',
		"message": str(e),
	}		

	return Response(predictions)


