This repository contains implementation and augmentation of NPS introduced in [1].
The code is based on NPS paper repository [2], as well [3] for the physics experiment.
For detailed analysis of the experiments see report.pdf. The repository also contains trained models and 
output images.

arithmetic

MNIST 

run commands in MNIST folder
	!pip install -r requirements.txt
	!pip install -e git+https://github.com/ncullen93/torchsample.git@ea4d1b3975f68be0521941e733887ed667a1b46e#egg=torchsample
and then run inside each experiment directory
	run.sh seed 

	six transformations(six_transMNIST):
	-evaluate NPS for 6 MNIST transformations
	-introduce exploration phase
	
	Composite Transformations(compMNIST, compMNIST_step):
	-two different implementaions of sequential NPS
	-take one image-digit and apply a composite transformation 
	-NPS manages to partially solve the task without overfitting in compMNIST_step
	and fully solve it by overfitting in compMNIST.
	
	Double Digit MNIST(catMNIST, catMNISTv2, catMNISTvseq):
	-take two or three digit-images as input and output a single, double or triple digit
	-NPS executes great on the first two setting but fails on the sequential one(catMNISTvseq)


	Customizing Transformations(cust_transMNIST)	
	-dataset contains image transformation of specific degrees and pixels
	-NPS manages to make 4 adjustable rules that can rotate or translate a digit-images
	at different scale
	

PHYSICS
	We integrate the NPS algotithm to the physics experiment as intoduce in paper [4]
	
	
[1] https://arxiv.org/pdf/2103.01937.pdf

[2] https://github.com/anirudh9119/neural_production_systems.git

[3] https://github.com/dido1998/CausalMBRL.git

[4] https://arxiv.org/abs/2107.00848	
