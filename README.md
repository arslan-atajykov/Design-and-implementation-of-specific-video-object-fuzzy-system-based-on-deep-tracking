# Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking
I have done this project as my final project for my bachelors degree during my studies in China.
This project focuses on the development and implementation of an advanced video object fuzzy system that leverages state-of-the-art object tracking and detection techniques. The primary objective is to enhance privacy and focus by selectively applying blur to specific class objects, such as persons, dogs, and cats, in a video stream.

First install requirements:
pip install -r requirements.txt
To run the code you can easily run : 
python3 gui_yolo.py

After you will have a gui window in which you can select the video you want to use, then select either target blur or roi select, then choose the blur rate and it will track and blur in realtime.
Overall  structure is <img width="625" alt="Screenshot 2023-12-29 at 17 45 52" src="https://github.com/arslan-atajykov/Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking/assets/65181705/23acc1dc-2fb4-481e-8ea7-72db0bc85cf7">


This is the main window 
<img width="745" alt="Screenshot 2023-12-29 at 17 46 30" src="https://github.com/arslan-atajykov/Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking/assets/65181705/997f4fc8-de96-4982-bc33-0f1dd87762c5">
After choosing the function target blur you will see this window 
<img width="892" alt="Screenshot 2023-12-29 at 17 46 36" src="https://github.com/arslan-atajykov/Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking/assets/65181705/7dfe320b-f53c-4138-a12e-52d28b4a684c">

The real time process
<img width="898" alt="Screenshot 2023-12-29 at 17 48 14" src="https://github.com/arslan-atajykov/Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking/assets/65181705/e6d6a98c-6b41-444e-be87-7839b85f40c3">

And the second function is ROI select :
<img width="832" alt="Screenshot 2023-12-29 at 17 48 57" src="https://github.com/arslan-atajykov/Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking/assets/65181705/ce1d4880-5659-40e7-87e2-d37f93a55593">

And the result: 
<img width="835" alt="Screenshot 2023-12-29 at 17 49 18" src="https://github.com/arslan-atajykov/Design-and-implementation-of-specific-video-object-fuzzy-system-based-on-deep-tracking/assets/65181705/36b8c669-1a22-4dd2-bbbe-1bcb5f578fe2">

