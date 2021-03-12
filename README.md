# CineML: Perfect Human

Through the process of training with StyleGAN2, I look into a 13-min short film in Black and White [The Perfect Human](https://www.youtube.com/watch?v=XqQeSf24phU) directed by Jørgen Leth in 1967 and re-examine the idea of Perfect Human that was previous portrayed as a middle class Danish couple performing everyday rituals.

## Pre-processing of the Dataset

Awaring the fact that the GAN aesthetics has been square and requiring 2^n as the number of pixels each side of the square need to have, and remain the authenticity of the footages, I choose not to crop but refill the upper and bottom part with black pixels. YouTube downloader was used to get the video file in mp4 and then the pre-process of the video was done with Premiere.

After exporting the video from Premiere, I started the Google Colab Notebook. After identifying the right file location for exported video in Google folder. I ran ffmpeg to extract frames from the clip as the formation of the dataset:

```python
!ffmpeg -i /content/drive/MyDrive/cineML-Perfect-Human/Jørgen_Leth-The_Perfect_Human_1967_360p_square.mp4 -r 12 /content/drive/MyDrive/stylegan2-colab/cineML-Perfect-Human/stylegan2/dataset/output-%04d.jpg
```

Note that the first file location following "-i" is where the exported video at. And the number following '-r' represents how many frame you are going to extract per second. As for the output addresses output-%04d.jpg is meant to generate filename such as "output-0001.jpg" Thus pre-calcuating the size of the image dataset is critial here to determine how many frames I need to extract per second.

Considering that the video itself lasted for a duration of 13m 04s, which is equivalent to 784 seconds in total. For a decent amount of images and a faster training process, instead of taking 24 frames per second, I chose 12 frames per second, which leads up to 9408 images in total.

## Training the Model

Utilizing the previous method of layering with artificial intelligence more familierly, I loaded raw images and started training:

```python
!python dataset_tool.py create_from_images ./dataset/the-perfect-human /content/drive/MyDrive/stylegan2-colab/cineML-Perfect-Human/stylegan2/raw_images
!python run_training.py --num-gpus=1 --data-dir=./dataset --config=config-f --dataset=the-perfect-human  --metrics=None
```

I ran the training program for 16h 33m 12s which results in the 709th iteration of the model. The output .pkl file would really set me in a good starting point for next phases of image generation.

## Generating Images
