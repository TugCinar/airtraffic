![Image](images/image.png)

airtraffic
==============================

forecast air traffic


Project description
------------


The objective of the code is therefore to forecast air traffic according to the base airport and the destination airport. To achieve this we will use the Prophet model developed by facebook (link to documentation: https://facebook.github.io/prophet/).
The results of the prediction will be readable in red on the graph displayed on the streamlit application.

Environment
------------

1. Create your virtual environment:
```conda create --name airtraffic```
2. Activatz your virtual environment:
```conda activate airtraffic```
3. Install Cookiecutter and streamlit with anaconda prompt :
```pip install Cookiecutter streamlit```

Repo
------------

Use Cookiecutter to clone folders

Launch the streamlit application  
------------
1. Use with anaconda prompt:
```streamlit run app.py ```
2. To use the application please select the origin and destination airports.
Some airports do not serve other airports, so here is a list of airports that contain data:
LGW-AMS
LGW-BCN
LYS-OPO 
LYS-ORY

3.Then select the date and the number of days you want to predict. The result is in red and you can see the different evaluations of the model

Modification
------------

you can modify the source code of app.py using local software





<p><small>Project based on the <a target="_blank" href="http://git.equancy.io/tools/cookiecutter-data-science-project/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
