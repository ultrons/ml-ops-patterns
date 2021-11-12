#FROM pytorch/torchserve:latest-cpu
FROM pytorch/torchserve:0.4.1-cpu

# install dependencies
RUN pip3 install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
RUN pip3 install --no-cache-dir  --force-reinstall -Iv grpcio
RUN pip3 install pyperclip
#RUN pip install torchserve torch-model-archiver torch-workflow-archiver


RUN pip install pytorch_lightning==1.4.9 
RUN pip install pytorch_forecasting

# copy model artifacts, custom handler and other dependencies
COPY ./forecast_handler.py /home/model-server/
COPY model.pt /home/model-server/

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=pt_tmp_forecast \
  --version=1.0 \
  --serialized-file=/home/model-server/model.pt \
  --handler=/home/model-server/forecast_handler.py \
  --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "pt_tmp_forecast=pt_tmp_forecast.mar", \
     "--model-store", \
     "/home/model-server/model-store"]

