# Export file in Docker Image

## Build/Run docker
### 1. ```docker build . -t test```
### 2. ```docker run -it test```

## get in Docker image and manually run CMD
### 1. ```docker run --rm -it --entrypoint bash test```
### 2. ```python main.py```

## check & export result file 2110999_TestResult.csv

### 1. ```docker cp <containerId>:/file/path/within/container /host/path/target```
