from ubuntu:20.04

RUN apt update
RUN apt install -y python
RUN apt install -y python3-pip

RUN pip install opencv-python-headless
RUN pip install torch torchvision
RUN pip install dominate
RUN pip install visdom
RUN pip install wandb
RUN pip install imagehash



COPY . .
CMD ["python3", "test.py", "--dataroot", "./datasets/apple2orange", "--name", "apple2orange_cyclegan", "--model", "cycle_gan", "--gpu_ids", "-1"]
# CMD ["python", "-c", "import os;print(os.getcwd())"]
# CMD pwd && ls