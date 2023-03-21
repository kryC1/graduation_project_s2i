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
#image_paths = [os.path.join('/static/images', f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

def home(request):
    if request.method == 'POST':
        my_data = request.POST.get('my_data')
        print(image_dir)

        item_list, image_path_list = start_process(my_data)

        #item_list = ["ali", "veli", my_data]
        #image_path_list = ["images/general/2360194369_d2fd03b337.jpg", "images/general/1095980313_3c94799968.jpg"]

        # The input text is input_field.
        # we can send output through context to fronted.

        context = {
            'items': item_list,
            'images': image_path_list
        }

        return render(request, 'stories2image/home.html', context)
    else:
        return render(request, 'stories2image/home.html')
