
# [CVIP 25]SADA: Unsupervised Camouflaged Object Detection using Foundation Models and Cue-Guided Refinement

![Framework](figure/CVIP_Architecture(1).png)

#Installation 
absl-py==2.1.0
backcall @ file:///home/conda/feedstock_root/build_artifacts/backcall_1592338393461/work
backports.functools-lru-cache @ file:///home/conda/feedstock_root/build_artifacts/backports.functools_lru_cache_1702571698061/work
cachetools==5.5.2
certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1725278078093/work/certifi
charset-normalizer==3.4.2
cycler==0.11.0
debugpy @ file:///home/conda/feedstock_root/build_artifacts/debugpy_1649586340600/work
decorator @ file:///home/conda/feedstock_root/build_artifacts/decorator_1641555617451/work
einops==0.4.1
entrypoints @ file:///home/conda/feedstock_root/build_artifacts/entrypoints_1643888246732/work
et-xmlfile==1.1.0
filelock==3.12.2
fonttools==4.38.0
fsspec==2023.1.0
google-auth==2.40.3
google-auth-oauthlib==0.4.6
grpcio==1.62.3
huggingface-hub==0.16.4
idna==3.10
imageio==2.31.2
importlib-metadata==6.7.0
ipykernel @ file:///home/conda/feedstock_root/build_artifacts/ipykernel_1666723258080/work
ipython @ file:///home/conda/feedstock_root/build_artifacts/ipython_1651240553635/work
jedi @ file:///home/conda/feedstock_root/build_artifacts/jedi_1696326070614/work
jupyter_client @ file:///home/conda/feedstock_root/build_artifacts/jupyter_client_1673615989977/work
jupyter_core @ file:///home/conda/feedstock_root/build_artifacts/jupyter_core_1658332345782/work
kiwisolver==1.4.5
Markdown==3.4.4
MarkupSafe==2.1.5
matplotlib==3.5.3
matplotlib-inline @ file:///home/conda/feedstock_root/build_artifacts/matplotlib-inline_1713250518406/work
nest_asyncio @ file:///home/conda/feedstock_root/build_artifacts/nest-asyncio_1705850609492/work
networkx==2.6.3
numpy==1.21.6
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
oauthlib==3.2.2
opencv-python==4.7.0.72
openpyxl==3.1.2
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1696202382185/work
pandas==1.3.5
parso @ file:///home/conda/feedstock_root/build_artifacts/parso_1712320355065/work
pexpect @ file:///home/conda/feedstock_root/build_artifacts/pexpect_1706113125309/work
pickleshare @ file:///home/conda/feedstock_root/build_artifacts/pickleshare_1602536217715/work
Pillow==9.5.0
prompt_toolkit @ file:///home/conda/feedstock_root/build_artifacts/prompt-toolkit_1727341649933/work
protobuf==3.20.1
psutil @ file:///home/conda/feedstock_root/build_artifacts/psutil_1666155398032/work
ptflops==0.7.3
ptyprocess @ file:///home/conda/feedstock_root/build_artifacts/ptyprocess_1609419310487/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl
pyasn1==0.5.1
pyasn1-modules==0.3.0
Pygments @ file:///home/conda/feedstock_root/build_artifacts/pygments_1700607939962/work
pyparsing==3.1.4
pysodmetrics==1.4.0
python-dateutil @ file:///home/conda/feedstock_root/build_artifacts/python-dateutil_1709299778482/work
pytz==2025.2
PyWavelets==1.3.0
PyYAML==6.0
pyzmq @ file:///home/conda/feedstock_root/build_artifacts/pyzmq_1652965483789/work
requests==2.31.0
requests-oauthlib==2.0.0
rsa==4.9.1
scikit-image==0.19.2
scipy==1.7.3
six @ file:///home/conda/feedstock_root/build_artifacts/six_1620240208055/work
tabulate==0.9.0
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorboardX==2.5.1
tifffile==2021.11.2
timm==0.6.12
torch==1.13.1
torchvision==0.14.1
tornado @ file:///home/conda/feedstock_root/build_artifacts/tornado_1656937818679/work
tqdm==4.64.1
traitlets @ file:///home/conda/feedstock_root/build_artifacts/traitlets_1675110562325/work
typing_extensions==4.7.1
urllib3==2.0.7
wcwidth @ file:///home/conda/feedstock_root/build_artifacts/wcwidth_1699959196938/work
Werkzeug==2.2.3
zipp==3.15.0
# Paper
[SADA]()
# Download SADA Benchmarks Dataset.
- COD10K: [google](https://dengpingfan.github.io/pages/COD.html) 
- CAMO: [google](https://sites.google.com/view/ltnghia/research/camo) 
- NC4K: [google] (https://github.com/JingZhang617/COD-Rank-Localize-and-Segment)
- CHAMELEON: [google] (https://drive.google.com/drive/folders/1LN4sP2DRtWcWHcgDcaZcWBVZfoJKccJU?usp=drive_link) 

# Generate Pseudo labels.
- We divided the dataset in to train and test. We took 3040 images from COD10K and 1000 images from CAMO in to the training set.
- We generated pseudo labels out these images using SAM.
- We have provided code to generate pseudo labels and store the best mask.

# Cue Region extraction
- Insert the proper file name and run `python generate_cue.py` 

# Training
- Run `python train.py` 

# Test and Evaluate
- Change your path and filename accordingly
- Run python test.py
- results.txt will be stroed in PySODEvalToolkit

# Experimental Results
![Result](figure/Result.png)

# Acknowledgement
[SAM- Segment-Anything](https://github.com/facebookresearch/segment-anything)
[PCOD](https://arxiv.org/abs/2408.10777)

