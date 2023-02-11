# Train CIFAR10 with PyTorch

The Repo contains the code for training CIFAR10 Classes data using PyTorch.

   <img src = "https://user-images.githubusercontent.com/60026221/218258957-7eaf33bf-172d-482e-b4f2-556c5cb5e90f.jpeg" width = 480px height = 340px>
     
- Based on this repo, further there will be addition of different model architectures to train the data with lighter, heavier size models in models directory.
- Simply, it is a store room of storing different model architectures. 
     
* It has following files:
------------------------

**Models**: It is a directory where we are gonna store all of our model architectures defined with a model name having an extension of .py.

**main.py**: This is where we can train our model with data to make it learn some thing and make it ready for usage of whatever cases we need  😜 (for prediction/production). 

**utils.py**: This is where we load data, perform data augmentations, do operations on data, plot the curves such as training & testing performances of model, mis-classified images, GRADCAM  🤔 Diagnoistic images (it helps to see how & where model is learning) (sample as below). 

* GRADCAM sample: 

   ![grad_cam](https://user-images.githubusercontent.com/60026221/218259556-97a240d2-9303-4f29-a798-048f74e04e90.JPG)


* Currently, making the model learn and all other things are going to be happen by importing classes and methods from files. The reference for implementing this is at : EVA/Assignment7/Jupyter notebook.


** Later this will happen: 
😎 😎 😎
## Training with CLI 
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```


Improvement Plan : To make it accessible via CLI (COMMAND LINE INPUTS/ARGUMEMNTS), as this is a base-repo for the next architectures to be made.
