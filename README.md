# CineML: Perfect Human

> Certain machine produced forms are the most perfect forms of our period.
> - G.Metzger, Auto-Destructive Art Machine Art Auto-Creative Art

This project is done in the Digital + Media studio class DM-7102 Stuio/Sem 2, under the guidance from professors Adela Goldbard, Alejandro Borsani and Mark Cetilia at Rhode Island School of Design.

Through the process of training with StyleGAN2, I look into a 13-min short film in Black and White [The Perfect Human](https://www.youtube.com/watch?v=XqQeSf24phU) directed by Jørgen Leth in 1967 and re-examine the idea of Perfect Human that was previous portrayed as a middle class Danish couple performing everyday rituals. 

Later this remake of computer generated moving images are transformed from a digital recreation to an analog artifect of celluoids, which is eventually played on a 16mm film projector as a film installation.

## Who is the Perfect Human?
What does it mean to be human? Or perfect?

What qualities does a human hold?

Does it represent a particular race? Sex? Sexuality? Ideology? Or cultural representation?

## Why Perfect Human?

To some extent, every film is stagged. The original film of The Perfect Human by Leth was too obviously staged in a boundless space. It was filming about a middle-class Danish couple perforing everyday rituals as if they are subjects in a zoom. The film itself examines human behavior in a "suave, pseudo-scientific" way.

It is an unrealistic environment to portray the non-existing perfect human in a bright demonstration room with dissection lights. White background successfully situates the concept of perfect human far from the reality.

To better contextualize the work, the year of release (1967) is also parallel with the prevalence of color TV in the 1960s and educational films that are used for various purposes.

Also a side note: The Perfect Human premiered at the Carlton cinema and was shown before Jean-Luc Godard's La Chinoise.


## Why Machine Learning?
And the reason I intentionally choose the 360p low-resolution video found on YouTube is to create an ambiguous and abstract remake that complicates the contested notion of perfect human. The computer-generated fakes also played a crucial part in generalizing what perfect human means.

I intend to challenge the concept of the perfect human with the intervention of GAN, in this case, StyleGAN2, which is commonly used for producing an unlimited number of portraits of fake human faces.


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

## Midterm Installtion
Previously, I envisioned the work to be installed at the entrance of CIT 4th Floor, since the space had what it takes to match the aesthetics of emptiness that appeared in the original film by Leth. Due to the size of pole doesn't match with the projector and speaker mounts etc, I had to move the installation to the Build Space also on the 4th floor of CIT building.

As for why I chose to present this through video installtion is because the projection will ease the crispness of pixels and blur the images with ambiguity as one of the reasons. Also historically, 1960s has been the years where there were a lot of "educational videos" being produced with similiar aesthetics.

![installation.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/Installation.png)

## Digital-to-Analog Conversion
Since the film remake with GAN still remains dream-like and film-like qualities, and has gone through processes from shooting directly in the set with 16mm film camera, being converted into digital file and uploaded on YouTube, to being downloaded and taken half of the frames as dataset, and training with GAN. Yet as digitalization becomes inevitably prevailing for longer preservation, I wonder if there is a digiatl-to-analog conversion to make artifacts that was originally computer-generated, and at the same time, it blurs general ideas that film captures the reality and how light leaves traces on chemicals. I intend to challenge these perceptions by transfering the digital video onto transparent film that can be projected with a 16mm film proojector.

## Why Print?
What is a film? Is it a roll of filmstrips or a video found online?

Over the course of decades ever since the release of the original film, there have been several iterations of the remake, either by the director Leth himself, or various filmmakers with different intentions. 

This process is challenging the authentication of a transitional way of seeing (film and projector), what you see might not be real even if it's from film.

### Creating 16mm Template
![template.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/template.png)
After several tests with plotted templates found online, I corrected the right spacing and margins in the InDesign files, with rulers and grids as guides, which is super helpful for the next part of importing 23,520 frames. This template made the testing such easier during the early stages. Because of the transparent sheets that I have easy access to are letter size, the 16mm template that I'm creating is also in size of 8.5 * 11 inches.

### Importing Frames
Once the template is plotted in InDesign by pressing "shift" + "arrows" to create images boxes in grid, go to "File > Place..." to select multiple frames. Then by clicking on the placeholder boxes we create, I assign source images to each of the frame on the super 16mm template. In order to adjust size of each source image to match with its placeholder, go to "Object > Fitting" and select either "Fit Content Proportionally" or "Content-aware Fit."

Note: "Fit Content Proportionally" means the whole image will be fit into the box with its original proportion, while "Content-aware Fit" means some of the images might be cut off.

However, when it comes to 23,520 frames, it is impossible to click on each of the box to place the still images. That was when  [ImageToCSV.jsx](https://creativepro.com/downloads/forcedl/imagesToCSV104.jsx) script file for InDesign comes in and saved my time. To install the scripts, all I have to do is to go to the script panel "Window > Untilities > Scripts" and then find the "User" folder under the scripts and click on "Reveal in Folder" button. Once the Finder window pops up, drag the script file into that folder and the import process is complete. Then by clicking on the ImageToCVS script and select the folder of choice and, a csv file of the image list will be generated.

Once the list of images is successfully created, we will open up the Data Merge Panel by going to "Window > Utilities > Data Merge," choose "Select Data Source" and look for the cvs file that was just generated. And the next step is to select one graphic frame and then click on the word "images" on the data merge panel. Finally we click on the "Create Merged Document" button and adjust spacing specs on the dialog box. Specs listed below:

Spacing | Distance
--------|----------
Top Margin | 12.779 mm
Bottom Margin | 7.93 mm
Left Margin | 0.7276 in
Right Margin | 0.3719 in
Arrange by | Columns First
Spacing Between Columns | 0.2995 in
Spcaing Bewteen Rows | 0.446 mm

Graphic frame (first image on sheet) transform specs below:

Spacing | Distance
--------|----------
X | 0.7376 in
Y | 12.779 mm
W | 0.4005 in
H | ~~7.175 mm~~ 7.621mm

#### NOTE: MAKE SURE THERE'S ONLY ONE IMAGE FRAME IN INDESIGN FILE...

### Printing on Film
The next is to test transparencies that are generally used for either inkjet printers and laser printers.

The major difference between these two type of printers is that, inkjet printers rely on dye or pigment-based ink to operate, while, laser printers, use toner to produce text and images. I have compared its quailty in prints and in projections below. Naturally inkjet has more bright and vivid colors and laser printers work better with black and white. The CMYK color model is quite obvious despite the darker quality in the laser prints, which mimics how cathode-ray tube works.

Side note: Color CRTs have three cathodes: one for red, green and blue. In color devices, an image is produced by controlling the intensity of each of three electron beams, one for each additive primary color (red, green, and blue) with a video signal as a reference.


### Laser Cutting
Once the template is set, I was also advised by Stephen to create a jig as same size as the laser cutting bed. With demension of 18 * 32 inches, the jig is planned to hold 3 sheets at a time in order to have a stable frame. After testing out that MDF (Medium Density Fiberboard) at risd:store 3D was too thick (1/2 in) to have the laser pointer moving around frames, I chose chipboards instead, despite the fact that it also needs additional weight to be held flat inside the machine.

And the process of crafting and making is filled with human errors despite the digital process of making. The parameters in the network training processes that directly affect the realness/closeness to the original data feed. The millimeters of error margin from printing on transparencies and placing on the laser cutter bed. The printed outcome is never perfect.

### Splicing
After FAV department kindly borrowed me a splicer, I was able to laser cut and splice toegther for earlier completion time. For each sheet it takes about 10 minute to cut and 10 minute to splice.

![splicer.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/splicer.png)


## Still Images

![man.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/man.png)

![eye.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/eye.png)

## Documentation Video
It was shot directly at the Steenbeck screen with a Sony a7 and a 50mm prime lens. A full video can be found here: [The Perfect Human](https://youtu.be/ukLvM74W9g4)

## Final Installation
I was having an extremely difficult time deciding what contents or part of the process I want to show, and where to install the work. Until the very last minute I landed on the idea of showing a two-channel film and video installation with uncut transparent sheets and printed-and-cut filmstrips on the moveable wall. There are two screening planned in the afternoon of the gallary time at 2:30pm and 5pm. Luckily the Kodascope 16mm film projector I found on eBay had no hiccup during th screenings.

![final.png](https://github.com/Belphe13/cineml-perfect-human/blob/master/final.png)

## Next Steps
1. Burn some filmstrips and document it
2. Find a better splicer like 16mm Guillotine (Ciro) M.2T Splicer Fixed Pins
2. Get a hold of a Kodak Pagent sound projector for sound testing
3. Use the inkjet process on something I filmed myself

## References
* [Jørgen Leth - The Perfect Human, 1967](https://www.youtube.com/watch?v=XqQeSf24phU)
* [FILM REVIEW; A Cinematic Duel of Wits For Two Danish Directors](https://www.nytimes.com/2004/05/26/movies/film-review-a-cinematic-duel-of-wits-for-two-danish-directors.html)
* [Import a Folder Full of Pictures, One Per Page](https://creativepro.com/import-folder-full-pictures-page/)
* [ImagesToCSV.jsx](https://creativepro.com/downloads/forcedl/imagesToCSV104.jsx)
* [How to Install Scripts in InDesign](https://creativepro.com/how-to-install-scripts-in-indesign/)
* ...

## Acknowledgement
I want to say huge thank you to everyone is D+M, the faculties at D+M for pushing me to work outside of my comfort zone, and my amazing cohort for their care and support. Especially to Nora and Stephen, I don't think I could ever complete this without your help.

Special thanks to professor Africanus Okokon from FAV who has been informing my research tremendously and guiding me through my early stages of experimentations.

Last but not least, Yutong, my partner and a future curator, for watching our pup Ollie when I'm away in PVD, borrowing a black and white laser printer from our friend, walking through my work conceptually and helping with the final installation.
