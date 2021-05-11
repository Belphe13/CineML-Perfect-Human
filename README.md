# CineML: Perfect Human

Through the process of training with StyleGAN2, I look into a 13-min short film in Black and White [The Perfect Human](https://www.youtube.com/watch?v=XqQeSf24phU) directed by Jørgen Leth in 1967 and re-examine the idea of Perfect Human that was previous portrayed as a middle class Danish couple performing everyday rituals.

## Why Perfect Human?

To some extent, every film is stagged. The original film of The Perfect Human by Leth was too obviously staged in a boundless space. It is an unrealistic environment to portray the non-existing perfect human. White background successfully situates the concept of perfect human far from the reality.

And the reason I intentionally choose the 360p low-resolution video found on YouTube is to create an ambiguous and abstract remake that complicates the contested notion of perfect human. The computer-generated fakes also played a crucial part of generalizing what perfect human means.


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

## Generating Images & Videos
Without doing anything else, the training results also generated screenshots that give a preview of which stage of "fakes" the network is at.

![fakes010330.jpg](https://github.com/Belphe13/cineml-perfect-human/blob/master/fakes/fakes010330.jpg)

Looking at these previews of generated outcomes in the grid, I cannot help but think about how the placement of images resembles the filmstrips, and each of the frames was isolated yet interconnected. Another feedback I got was related to the similarity bewteen the grid and building windows and how each of the "windows" is like encapsulation of people's life on display. Just like how the short film was observing how a couple 'function' in a blank studio setting as if they are subjects to be viewed.

So I decided to create an short animation with each results generated by GAN, again, using FFmpeg. The only requirement is to create a new folder filled in with copies of fake results then rename each of images to make sure that the file names are in sequence with the increment of 1. Then type in the command below:

```python
!ffmpeg -r 6 -i fakes%05d.jpg -vcodec libx264 -pix_fmt yuv420p fakes-15s.mp4
```

This will output a video composed of fake results. Note that "-r" marks the frame rate of the image. In this case, "-r 6" means that for every second there will be 6 images, and then with around 90 images, it will render a 15s long video.

The generated video can be viewed here: [Fake Results for the Perfect Human](https://vimeo.com/522684582).


## Image Sorting
After running the python command to generate 500 images from the trained network., which takes 1m 22s in Google Colab. I used image sorting software calledn [PicArrange](https://apps.apple.com/app/picarrange/id1530678223) developed by Kai-Uwe Barthel to rearrange these 500 images by similarity. At the same time, I manually broke down cuts of the short film and organized in a slides. Then the re-grouped these generated images with labels in order to map with the original film.

## Remapping
Remapping was time consuming because of a "manual" hash table I created for lookup the right sequences. This is where the process can be improved for accuracy and faster production. Below is the link to my attempt of remake via machine learning. 

[CineML: The Perfect Human](https://vimeo.com/533367370)

## Installtion
Previously, I envisioned the work to be installed at the entrance of CIT 4th Floor, since the space had what it takes to match the aesthetics of emptiness that appeared in the original film by Leth. Due to the size of pole doesn't match with the projector and speaker mounts etc, I had to move the installation to the Build Space also on the 4th floor of CIT building.

As for why I chose to present this through video installtion is because the projection will ease the crispness of pixels and blur the images with ambiguity as one of the reasons. Also historically, 1960s has been the years where there were a lot of "educational videos" being produced with similiar aesthetics.

![installation.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/Installation.png)

## Digital-to-Analog Conversion
Since the film remake with GAN still remains dream-like and film-like qualities, and has gone through processes from shooting directly in the set with 16mm film camera, being converted into digital file and uploaded on YouTube, to being downloaded and taken half of the frames as dataset, and training with GAN. Yet as digitalization becomes inevitably prevailing for longer preservation, I wonder if there is a digiatl-to-analog conversion to make artifacts that was originally computer-generated, and at the same time, it blurs general ideas that film captures the reality and how light leaves traces on chemicals. I intend to challenge these perceptions by transfering the digital video onto transparent film that can be projected with a 16mm film proojector.

### Research Process
Research...

### Creating Super 16mm Template
![]()
After several tests with plotted templates found online, I corrected the right spacing and margins in the InDesign files, with rulers and grids as guides, which is super helpful for the next part of importing 23,520 frames.

### Importing Frames
Once the template is plotted in InDesign, go to "File > Place..." to select multiple frames. Then by clicking on the placeholder boxes we create, I assign source images to each of the frame on the super 16mm template. In order to adjust size of each source image to match with its placeholder, go to "Object > Fitting" and select either "Fit Content Proportionally" or "Content-aware Fit."

Note: "Fit Content Proportionally" means the whole image will be fit into the box with its original proportion, while "Content-aware Fit" means some of the images might be cut off.

### Printing on Film
The next is to test transparencies that are generally used for either inkjet printers and laser printers.

The major difference between these two type of printers is that, inkjet printers rely on dye or pigment-based ink to operate, while, laser printers, use toner to produce text and images. I have compared its quailty in prints and in projections below. Naturally inkjet has more bright and vivid colors and laser printers work better with black and white. The CMYK color model is quite obvious despite the darker quality in the laser prints, which mimics how cathode-ray tube works.

* Color CRTs have three cathodes: one for red, green and blue. In color devices, an image is produced by controlling the intensity of each of three electron beams, one for each additive primary color (red, green, and blue) with a video signal as a reference.


### Laser Cutting


