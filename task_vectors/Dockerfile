command: >
  bash -c '
  set -e &&
  echo "Environment Variables:" &&
  echo "HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN" &&
  echo "LLAMA_DIR=$LLAMA_DIR" &&
  
  # conda環境作成を確認
  if [ ! -d "/opt/conda/envs/icl_task_vectors" ]; then
    echo "Creating conda environment..." &&
    conda create -n icl_task_vectors python=3.10 -y &&
    
    echo "Installing PyTorch..." &&
    conda install -n icl_task_vectors -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia &&
    
    echo "Installing data science packages..." &&
    conda install -n icl_task_vectors -y scipy scikit-learn matplotlib pandas -c conda-forge &&
    
    echo "Installing Jupyter..." &&
    conda install -n icl_task_vectors -y jupyter jupyterlab -c conda-forge &&
    
    echo "Installing transformers and datasets..." &&
    conda install -n icl_task_vectors -y transformers datasets -c huggingface -c conda-forge &&
    
    echo "Installing utility packages..." &&
    conda install -n icl_task_vectors -y requests tqdm python-dotenv schedule -c conda-forge &&
    
    echo "Installing Git..." &&
    conda install -n icl_task_vectors -y git -c conda-forge &&
    
    echo "Installing pip packages..." &&
    conda run -n icl_task_vectors pip install sentencepiece bitsandbytes nltk accelerate;
  else
    echo "conda environment already exists";
  fi &&
  
  echo "Ensuring LLaMA directory exists..." &&
  mkdir -p $LLAMA_DIR &&
  
  echo "Starting JupyterLab..." &&
  conda run -n icl_task_vectors jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token="Yuka0124"
  '