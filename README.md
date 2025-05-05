# ece272b-homework-3-variational-autoencoder-solved
**TO GET THIS SOLUTION VISIT:** [ECE272B Homework 3-Variational Autoencoder Solved](https://www.ankitcodinghub.com/product/ece272b-homework-3-variational-autoencoder-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100180&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ECE272B Homework 3-Variational Autoencoder Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Introduction

In this homework, we will continue working with convolution neural nets, but use them on appli- cation of data encoding and decoding, and we will also have a glance at data generation.

In this homework, we will explore Autoencoder model, more specifically, Variational Autoencoder. From a very abstract view, Variational Autoencoder maps the input data into the parameters of a probability distribution, such as the mean and variance of a Gaussian distribution. This approach produces a continuous, structured latent space, which is useful for data compression and image generation. As usual, you will use tensorflow library to build a complete pipeline for data preprocessing, model building and training, and result evaluation. You will also utilize free GPU/TPU resources available on Goolge Cloud for speeding up the training.

Data Set

We will use tf.keras.datasets module to download and load our datasets. We will be using two datasets: MNIST and CIFAR-10.

MNIST Dataset

The MNIST dataset has a training set of 60000 examples, and a test set of 10000 examples. The dataset has ten classes, each associates with a digit. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. [2].

Here’s the code snippet for loading the dataset:

<pre>(train_images, training_labels), (test_images, test_labels) = \
        tf.keras.datasets.mnist.load_data()
</pre>
CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60000 32×32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The class labels are airplane, auto- mobile, bird, cat, deer, dog, frog, horse, ship and truck. [1].

Here’s the code snippet for loading the dataset:

<pre>(train_images, training_labels), (test_images, test_labels) = \
        tf.keras.datasets.cifar10.load_data()
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Problem Formulation

You must perform the following guiding tasks and write down answers to the following questions as prompted in the notebook:

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
<div class="column">
MNIST VAE

<ol>
<li>(2 pts) Preprocess the data set. Show one image for each class from the MNIST training dataset.</li>
<li>(4 pts) Complete the code for defining the decoder net in CVAE class (2pts). What is the input shape to the first convolution transpose layer in the decoder? Explain how you decided this input shape. Show the calculation in terms of the img_size, num_convolutions, and strides used in the deconv layers (2pts).</li>
<li>(2 pts) What is the last layer of the decoder net in CVAE class? Explain how you decided the number of filters for this layer.</li>
<li>(2 pts) Read the function definition of log_normal_pdf. We know the probablity density function (PDF) of a normal distribution is:</li>
</ol>
2 1 −1(x−μ)2 f(xi;,)=√e2 σ

2π

where μ is mean, and σ is standard deviation. If we take log of the above equation, we get

the log PDF of the normal distribution:

log(f(xi;μ,σ2))=−ln− 1ln(2π)− 1(x−μ)2 22σ

Note that log_normal_pdf is given a sample x, the mean μ, and the log variance logvar of the distribution. Please derive the formula in tf.reduce_sum() in log_normal_pdf’s function return. (Hint: variance is the square of σ ).

<ol start="5">
<li>(2 pts) As described in this blog, to generate a sample for the decoder during training, you can sample from the latent distribution defined by the parameters outputted by the encoder, given an input observation, i.e, z ̃ ∼ q(z|x). However, this sampling operation creates a bottleneck because backpropagation cannot flow through a random node.
Refer to reparameterize function in CVAE class. Explain what is the Reparametrization Trick and how it enables the backpropagation.
</li>
<li>(2 pts) Recall from class that VAE has two optimization objectives:

(a) Maximize the log likelihood log p(x|z), or minimize the reconstruction error.

(b) Minimize the KL divergence of the approximate from the true posterior: DKL(log q(z|x)|| log p(z)). Describe which part in function compute_loss describes objective (a) and which part de-

scribes (an estimation of) objective (b)?

Hint: We can use Monte Carlo to estimate the KL divergence of continuous distributions when there’s enough data for sampling, i.e, N is large:
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
DKL(log q(z|x)|| log p(z)) ≈ N

<ol start="7">
<li>(3 pts) Complete function generate_and_display_images (2pts). Note that we use CVAE’s sample function to obtain the decoded images from latent space. Explain why we set apply_sigmoid == True when decoding (1pt). (Hint: Think of why we need to apply sig- moid on the decoder’s outputs.)</li>
<li>(2 pts) Plot the encoder and decoder architecture as a flow chart diagram with shape speci- fications below. Are the input and output shapes match your design?</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
1 􏰀N q ( z | x )

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
<div class="layoutArea">
<div class="column">
i

</div>
<div class="column">
log( p(z) )

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
9.

10. 11.

2

1.

2.

3

1. 2.

</div>
<div class="column">
(3 pts) Call the completed main_train_loop function. Show the following in the output console when training:

(a) Time used for each epoch training.

(b) Average ELBO (the negation of the loss value) on the test dataset for each epoch.

(c) Thereconstructedimagesofthesampledtestdataaftereachepoch(Usegenerate_and_display_images). (3 pts) Complete the function get_allClass_encodings and plot_latent_space. Visualize

the latent space. Is there a clear boundary between the classes?

(2 pts) Complete the function walk_src_to_dst. Show the morphing process of ’walking’ from one class to another.

CIFAR-10 VAE

(8 pts) Repeat the above workflow on CIFAR-10 data set. Try to reuse functions already defined. (Preprocess and visualize data 1 pt, Training 5 pts, visualize latent space 1 pt, walk from two classes 1 pt)

(1 pt) Were you able to generate clear images with the CIFAR-10 data set? Can you think of any ways to improve the quality of the reconstructed images?

Grad/Extra Credits

(4 pts) Modify compute_loss to use Maximum Mean Discrepancy (MMD) as described in another blog (2 pts) and train a new model with the new loss function on CIFAR-10 (2 pts).

(5 pts) Repeat the visualization of latent space (1 pt) and interpolation between two classes (1 pt). Did you observe any difference before and after using the MMD loss (1 pt). Refer to the blog, explain how MMD loss differs from the original loss function (2 pt).

</div>
</div>
</div>
