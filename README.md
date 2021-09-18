# OptionA : Run model locally 

### 1. ```pip isntall -r requirements.txt```
### 2. Check the directory named result, make sure it's empty
### 3. After finishing installation of necessary module
### 4. ```python manin.py```
### 5. You'll see the model output in the directory named result


# OptionB : Export file from Docker container

## Build/Run docker
### 1. ```docker build . -t perform_model```
### 2. ```docker run -d -it perform_model```

## check container ID
type command ```docker ps -a``` on terminal than you'll see the container id, than copy the first four letters.

If your ID is da6455c5f597 than copy ```da64```.

## Use docker cp command to move model output into local host

### type command ```docker cp [your id]:./app/result/110999_TestResult.csv  C:/Users/user/Desktop/dinner_th/result``` in terminal 

### for example: ```docker cp [id]:./app/result/110999_TestResult.csv  C:/Users/user/Desktop/dinner_th/result```
