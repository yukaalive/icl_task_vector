# ベースイメージ
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniconda/bin:$PATH"

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y curl git && \
    rm -rf /var/lib/apt/lists/*

# Miniconda のインストール
RUN curl -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3.sh -b -p /opt/miniconda && \
    rm Miniconda3.sh && \
    conda init bash

# Miniconda の base 環境に JupyterLab をインストール
RUN /opt/miniconda/bin/conda install -y jupyterlab && \
    /opt/miniconda/bin/conda clean -afy

# 作業ディレクトリ
WORKDIR /home/yukaalive/2025workspace

# Jupyter Lab をデフォルトで起動する
CMD ["/opt/miniconda/bin/jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token='Yuka0124'"]
