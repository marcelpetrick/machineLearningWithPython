These instructions are for windows 10.
First install python-3.6.8-amd64.exe. Be sure to check the Path change message in the beginning and also click on removing the path restriction at the end of installation. 

Then install CUDA. Use the standard setting. 

Then unzip the CUDNN ZIP file and copy paste into the CUDA folder (copy from root to root). 

Next, make sure the CUDA bin foler (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin) is in the PATH (environment variables). 
ALSO MAKE SURE THE PYTHON PATH IS THERE AND IS CORRECT! Sometimes there can be a mix up between Roaming and Local

Next, type CMD in the windows start, and right click on it and run as administrator. 

When the command line opens, update pip by:

python -m pip install --upgrade pip

Then: pip install --upgrade --user numpy scipy pandas matplotlib sklearn jupyter tensorflow-gpu keras

If your computer doesn't have an NVIDIA GPU, run tensorflow (instead of tensorflow-gpu).


#################################
# state of the installed files: #
#################################
$ ls -lah
total 136K
drwxr-xr-x 1 kurs 197121    0 Jul  4 19:09 ./
drwxr-xr-x 1 kurs 197121    0 Jul  4 17:51 ../
drwxr-xr-x 1 kurs 197121    0 Jul  4 19:11 .git/
drwxr-xr-x 1 kurs 197121    0 Jul  4 18:26 .ipynb_checkpoints/
-rw-r--r-- 1 kurs 197121  987 Jul  3 17:56 Anleitung_Pouyan.txt
-rw-r--r-- 1 kurs 197121 3,9K Feb  5  2014 iris.csv
-rw-r--r-- 1 kurs 197121  70K Jul  4 19:04 IRIS.ipynb
-rw-r--r-- 1 kurs 197121  35K Jul  4 17:51 LICENSE
-rw-r--r-- 1 kurs 197121 8,8K Jul  4 19:09 README.md
