FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3
ARG SDK_VERSION=0.6.0

COPY . /root
WORKDIR /root

# Installs additional packages
RUN pip3 uninstall -y google-cloud-aiplatform
RUN pip3 install google-cloud-aiplatform tabulate
RUN pip3 install google-cloud-aiplatform-$SDK_VERSION.tar.gz
RUN pip3 install .

