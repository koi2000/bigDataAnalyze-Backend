import json

from django.http import HttpResponse, JsonResponse
from rest_framework.response import Response
# Create your views here.
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer

from bigdata.LSTM_poem.test import generateApi, gen_acrosticApi
from bigdata.utils.response.renderer import CustomRenderer
from bigdata.utils.response.result import ok_data
from bigdata.word2vec.test import query_nearestWord


def index(request):
    return HttpResponse("欢迎使用")

@api_view(('POST',))
def queryNearestWord(request):
    jsonData = json.loads(request.body)
    wordList = jsonData['wordList']
    num = jsonData['num']
    res = query_nearestWord(wordList, num)
    return ok_data(res)

@api_view(('POST',))
def genPoem(request):
    jsonData = json.loads(request.body)
    print(jsonData)
    res = generateApi(jsonData['text'])
    return Response(res)


@api_view(('POST',))
# @renderer_classes((CustomRenderer,))
def genAcrostic(request):
    jsonData = json.loads(request.body)
    print(jsonData)
    res = gen_acrosticApi(jsonData['text'])
    print(res)
    return Response(res)
