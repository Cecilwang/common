FROM ubuntu:18.04

MAINTAINER Sixue Wang <cecilwang95@gmail.com>

########## deb packages ##########
RUN apt update && apt install -y --no-install-recommends \
        git \
        lcov \
        make \
        # used by add-apt-repository
        software-properties-common \
        wget \
        && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

########## GCC ##########
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && apt install -y --no-install-recommends \
      gcc-9 \
      g++-9 \
      && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 900 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 900

########## Go ##########
RUN wget -qO- https://golang.org/dl/go1.16.3.linux-amd64.tar.gz | tar xzf - -C /usr/local
ENV PATH ${PATH}:/usr/local/go/bin:/root/go/bin

########## Bazel ##########
RUN go get github.com/bazelbuild/bazelisk && \
    ln -s /root/go/bin/bazelisk /root/go/bin/bazel
