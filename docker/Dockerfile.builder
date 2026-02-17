FROM intel/oneapi-basekit:latest
RUN apt-get update && apt-get install -y cmake build-essential git
WORKDIR /build
