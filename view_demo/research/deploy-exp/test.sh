torch-model-archiver -f \
  --model-name=pt_tmp_forecast \
  --version=1.0 \
  --serialized-file=./model.pt \
  --handler=./forecast_handler.py \
  --export-path=./model-store

torchserve \
     --start \
     --ts-config=./config.properties \
     --models \
     pt_tmp_forecast=pt_tmp_forecast.mar \
     --model-store ./model-store \
