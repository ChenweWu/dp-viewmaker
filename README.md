# DP-Viewmaker: Neural Obfuscation of Private Attributes in Medical Images
Chenwei Wu, Chunyu Wang

Umich Class project for EECS598 Biomedical AI with Prof Liyue Shen

Thanks Dr. Alex Tamkin at Stanford for providing the model codes for viewmaker networks.
Thanks Dr. Leo Anthony Celi, Dr. Luis Nakayama at MIT for providing the datasets and inspirations.


## 0) Background

Patient sensitive information protection has been a increasingly hot concern for
the medical machine learning community. Recent work has shown that medical
imaging datasets across multiple modalities (ophthalmology, radiology) are at risk
of adversary privacy attacks, posing a huge hindrance to public data sharing and
fair clinical AI. Yet recent work in differential privacy for medical images has
fallen short in either retaining downstream clinical utility or generalizability across
different domains. Hence we propose a neural de-identification framework that
works across different medical imaging modalities. Our DP-Viewmaker framework
obfuscates a medical image in an adversarial way, protecting it from autonomous
attacker and preserving its pathological features. Using gender as a proxy for
private attributes, our model achieved near perfect de-identification while retaining
high disease classification performance on retinal images and X-ray images.

## 1) Install Dependencies

We used the following PyTorch libraries for CUDA 10.1; you may have to adapt for your own CUDA version:

```console
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other dependencies:
```console
pip install -r requirements.txt
```

## 2) Running experiments

Start by running
```console
source init_env.sh
```

Now, you can run experiments for the different modalities as follows:

```console
python scripts/run_image.py config/image/pretrain_viewmaker_xray.json --gpu-device 0
```

This command runs viewmaker pretraining on the CheXpert X-Ray dataset using GPU #0. 

```console
python scripts/run_image.py config/image/pretrain_viewmaker_brset.json--gpu-device 0
```

This command runs viewmaker pretraining on the BR-SET Retinal Images dataset using GPU #0.

(If you have a multi-GPU node, you can specify other GPUs.)

```console
python downstream.py
```
This command runs downstream evaluation on BR-Set Image Classification based on ./dsconfig.json
```console
python ds_visuals.py
```
This command runs downstream visualizations on BR-Set Image De-identification

The `config` directory holds configuration files for the different experiments,  specifying the hyperparameters from each experiment. The first field in every config file is `exp_base` which specifies the base directory to save experiment outputs, which you should change for your own setup.

You are responsible for downloading the datasets. Update the paths in `src/datasets/root_paths.py`.

Training curves and other metrics are logged using [wandb.ai](wandb.ai)
