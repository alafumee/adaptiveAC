## Quick tutorial for using docker

### 1. Install docker
Make sure you have docker installed on your machine and your user is in the docker group. 

### 2. Build the docker image
To build the docker image, first modify in following in `build.sh` using your uid and gid (you can find them by running `id` in the terminal):
```Dockerfile
RUN groupadd -g $(gid) torchuser
RUN useradd -r -u $(uid) -g torchuser --create-home torchuser
```

Then run the following command in the terminal:
```bash
sh build.sh
```

### 3. Run the docker container
To run the docker container, run the following command in the terminal:
```bash
sh run.sh
```
For libraries that require GUI, you can run the following command:
```bash
sh run.sh gui
```