MODEL_NAME=pt_tmp_forecast
CUSTOM_PREDICTOR_IMAGE_URI="gcr.io/pytorch-tpu-nfs/${MODEL_NAME}"

docker build \
  --tag=$CUSTOM_PREDICTOR_IMAGE_URI \
  .
APP=local_$MODEL_NAME
docker stop $APP
docker run -t -d --rm -p 7080:7080 --name=$APP $CUSTOM_PREDICTOR_IMAGE_URI
sleep 20

curl http://localhost:7080/ping
