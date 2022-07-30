# Centimeter-Wave Free-Space Neural Time-of-Flight Imaging
### [Project Page](https://light.princeton.edu/publication/ghztof/) | [Paper](https://dl.acm.org/doi/10.1145/3522671) | [Pretrained ckpts]()

[Seung-Hwan Baek*](https://www.shbaek.com/),[Noah Walsh*](), [Ilya Chugunov](https://ilyac.info/), [Zheng Shi](https://zheng-shi.github.io/), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

If you find our work useful in your research, please cite:
```
@article{baek2022centimeter,
  title={Centimeter-Wave Free-Space Neural Time-of-Flight Imaging},
  author={Baek, Seung-Hwan and Walsh, Noah and Chugunov, Ilya and Shi, Zheng and Heide, Felix},
  journal={ACM Transactions on Graphics (TOG)},
  year={2022},
  publisher={ACM New York, NY}
}
```

## Requirements
This code is developed using Pytorch on Linux machine. Full frozen environment can be found in 'environment.yml', note some of these libraries are not necessary to run this code. 

## Data
In the paper we use [Hypersim RGB-D](https://github.com/apple/ml-hypersim) dataset as our training data. And they can be easily swtich to any other RGB-D datasets of your choice. See 'dataloader/' folder for more details. 

## Testing
To perform inference on real-world captures, please first download the pre-trained model from [here]() to 'ckpts/' folder, then you can run the 'inference.ipynb' notebook in Jupyter Notebook. The notebook will load the checkpoint and process captured sensor measurements located in 'captures/'. The reconstructed depth will be displayed within the notebook.

## Training
We include 'train.sh' for training purpose. 

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. 

## Questions
If there is anything unclear, please feel free to reach out to Seung-Hwan at shwbaek[at]postech[dot]ac[dot]kr.
