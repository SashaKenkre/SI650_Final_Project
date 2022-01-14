# FBI Missing Persons Search Engine
### Team: Sasha Kenkre and Bulgan Judgerkhuu

This is a final project for SI650/EECS 649 at UMSI for Fall 2021.
    
This project looks at the [FBI's Missing Person Database](https://www.fbi.gov/wanted/kidnap) and tries to improve upon the search capability. When we began this project, the FBI database could not properly yield results for descriptive searches. However, as we are nearing the end of the project, it seems they have updated their search capabilities. 

# Data Sources
* [FBI's Missing Person Database](https://www.fbi.gov/wanted/kidnap)

# Required Python Packages
* pandas==1.2.1
* numpy==1.19.5
* requests==2.25.1
* Flask==1.1.2
* Werkzeug==1.0.1
* nltk==3.5
* rank-bm25==0.2.1
* mtcnn==0.1.1
* Pillow==8.1.0
* scipy==1.6.1
* keras-vggface==0.6
* Keras-Applications==1.0.8
* pickle-mixin==1.0.2

If you have issues installing keras, check out this [StackOverflow post](https://stackoverflow.com/questions/68862735/keras-vggface-no-module-named-keras-engine-topology).

# How to Run the Program

**Step 1: Download all files and unzip static folder**

Download the files provided in this repo, unzip the folder, and make sure all files and folders are stored in the same place. Make sure to move to the correct folder location in your terminal where you've housed all these unzipped files.

**Step 2: Install packages**
```
$ pip3 install -r requirements.txt
```
You can install the provided requirements.txt file. You may need to adjust the code above for installing the packages depending on your OS. 

**Step 3: Run project_flask.py**
```
$ python3 project_flask.py
```
Run the project_flask.py file to get the below link. 

**Step 4: Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

Go to the link provided ([http://127.0.0.1:5000/](http://127.0.0.1:5000/)) in your terminal after running project_flask.py. This will give you access to the webpage to interact with the program and its search capabilities.

There are two functional search options options and a third that is a work in progress (you can upload an image and see that image, but the search results won't show yet). The functional search capabilities include:
1. Text based search
2. Drop down name search


