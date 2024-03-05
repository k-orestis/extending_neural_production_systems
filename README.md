
# Neural Production Systems (NPS) Implementation and Augmentation

This repository contains an implementation and augmentation of Neural Production Systems (NPS) as introduced in the paper by [XYZ](https://arxiv.org/pdf/2103.01937.pdf). The code is built upon the NPS paper repository ([GitHub](https://github.com/anirudh9119/neural_production_systems.git)), as well as [XYZ](https://github.com/dido1998/CausalMBRL.git) for the physics experiment. For a detailed analysis of the experiments, refer to `report.pdf`. The repository also contains trained models and output images.

## Contents

### Arithmetic

#### MNIST

In the `MNIST` folder, execute the following commands:

```bash
pip install -r requirements.txt
pip install -e git+https://github.com/ncullen93/torchsample.git@ea4d1b3975f68be0521941e733887ed667a1b46e#egg=torchsample
```

Then, run inside each experiment directory:

```bash
./run.sh seed
```

#### Six Transformations (six_transMNIST):

- Evaluate NPS for 6 MNIST transformations
- Introduce exploration phase

#### Composite Transformations (compMNIST, compMNIST_step):

- Two different implementations of sequential NPS
- Take one image-digit and apply a composite transformation
- NPS manages to partially solve the task without overfitting in `compMNIST_step` and fully solve it by overfitting in `compMNIST`.

#### Double Digit MNIST (catMNIST, catMNISTv2, catMNISTvseq):

- Take two or three digit-images as input and output a single, double, or triple digit
- NPS performs well on the first two settings but fails on the sequential one (`catMNISTvseq`)

#### Customizing Transformations (cust_transMNIST):

- Dataset contains image transformation of specific degrees and pixels
- NPS manages to create 4 adjustable rules that can rotate or translate digit-images at different scales

### Physics

We integrate the NPS algorithm into the physics experiment as introduced in [XYZ](https://arxiv.org/abs/2107.00848).

## References

[1] [NPS Paper](https://arxiv.org/pdf/2103.01937.pdf)  
[2] [NPS Paper Repository](https://github.com/anirudh9119/neural_production_systems.git)  
[3] [Physics Experiment Repository](https://github.com/dido1998/CausalMBRL.git)  
[4] [Physics Experiment Paper](https://arxiv.org/abs/2107.00848)


