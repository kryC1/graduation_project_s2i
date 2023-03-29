from django.shortcuts import render
from django.http import HttpResponse

import sys
sys.path.append("/home/kryc1/stories2image")
sys.path.append("/home/kryc1/stories2image/src/image/captioning")
from s2i_main import start_process

import os
sys.path.append("/home/kryc1/s2i_django/s2i_django")
import settings
image_dir = os.path.join(settings.BASE_DIR, 'stories2image', 'static', 'images')

def home(request):
    if request.method == 'POST':
        my_data = request.POST.get('my_data')
        print(image_dir)

        splitted_list, image_path_list, genres_list, keywords_new_list, keywords_org_list, summary_list = start_process(my_data)

        context = {
            'parts': splitted_list,
            'images': image_path_list,
            'genres': genres_list,
            'keywords_new': keywords_new_list,
            'keywords_org': keywords_org_list,
            'summaries': summary_list
        }

        return render(request, 'stories2image/home.html', context)
    else:
        return render(request, 'stories2image/home.html')
