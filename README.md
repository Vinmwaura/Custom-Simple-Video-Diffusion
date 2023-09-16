# Custom-Simple-Video-Diffusion

This is an attempt at implementing a Diffusion model for video generation by modifying a 2D U-Net to a 3D U-Net with additional changes to be able to support input of multiple sequential frames from a video and generate temporally coherent frames that when combined can be able to be saved as a gif or video and be played.

## How it works
This project utilises two models for generating a GIF of K frames:

1. Base model.

This utilises a Diffusion model to generate X frames. We load N sequental frames from a video dataset and starting from the initial frame we skip M frames in-between to obtain X frames which is then degraded by adding noise and then predicting the noise added to it.

![Diagram showing how base model works](https://github.com/Vinmwaura/Custom-Simple-Video-Diffusion/blob/main/Base%20Model%20Reconstruction.jpg)

2. Interpolation model.

This utilises a Cold-Diffusion model which reconstructs the image including an in-between frames from the generated X frames in the Base model to increase the frames to K frames. 

![Diagram showing how interpolation model works](https://github.com/Vinmwaura/Custom-Simple-Video-Diffusion/blob/main/Base%20Model%20Interpolation.jpg)

## Examples of generated videos.
### DashCam Video
Dataset Source: https://youtu.be/qyN5aLRnZ-E

![Gif of Dashcam](https://github.com/Vinmwaura/Custom-Simple-Video-Diffusion/blob/main/GIFS/DashCam.gif)

### Kitten Video
Dataset Source: https://www.youtube.com/watch?v=ftgcwsBqS0U

![GIF of Kittens playing](https://github.com/Vinmwaura/Custom-Simple-Video-Diffusion/blob/main/GIFS/Kitten.gif)

### Squirrel and Birds Video
Dataset Source: https://www.youtube.com/watch?v=C2EvpdSxOQg

![GIF of Squirrel and Birds](https://github.com/Vinmwaura/Custom-Simple-Video-Diffusion/blob/main/GIFS/Squirrel_and_Birds.gif)

