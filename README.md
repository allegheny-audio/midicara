# Midicara

Midicara is a software midi controller that outputs midi signals based on facial landmarks. These facial landmarks come from a standard developed by **i-bug** and can be found [here](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).

## Building

To build, start by downloading the `nix` package manager:

```sh
# FIXME
# Ubuntu
sudo apt-get install nix
# Fedora
sudo dnf install nix
# Arch
packman install nix
```

Jump into a nix shell to gain access to all of the libraries and other outputs:

```sh
> nix shell
```

then invoke the build process by using `cmake` from the repository base directory:

```sh
> mkdir -p build
> cd build
> cmake ..
> cmake --build .
```

## Running

Before running,

1. Ensure you have a *shape predictor* file downloaded. `dlib` has one downloadable on their website. This is necessary for the face detection model to work. You will need to invoke this on the program start.
    ```sh
    > # download zipped .dat file from dlib.net
    > curl -X "GET" "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" --output shape_predictor_68_face_landmarks.dat.bz2
    > # unzip the bz2 file
    > bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    ```
2. Have a midi-capable device connected to your computer, so that midi ports can be enumerated properly. It does not need to be connected to an output, only to a device capable of receiving midi.
3. Make sure the camera you are using (webcam or not) has sufficient lighting such that your face, or the face of the user, will be visible with appropriate contrast. Avoid positioning the camera towards bright background light, as this will darken the subject on the camera.

Now that you are ready to run, you may invoke the executable:

```sh
> cd build
> ./feed /path/to/shape/predictor
```

A terminal interface will appear with various steps of configuration available.

# References

* [1] Alves-Pinto, Ana, et al. "The case for musical instrument training in cerebral palsy for neurorehabilitation." Neural plasticity 2016 (2016).
* [2] Bradski, G. "The OpenCV Library." Dr. Dobb's Journal of Software Tools, vol. 4, 2000, pp. N/A.
* [3] Frid, Emma. "Accessible digital musical instruments—a review of musical interfaces in inclusive music practice." Multimodal Technologies and Interaction 3.3 (2019): 57.
* [4] “Hands Free, Breath-Powered Instrument - Music for All.” Jamboxx, 12 Dec. 2019, www.jamboxx.com/. 
* [5] Davis E. King. Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research 10, pp. 1755-1758, 2009.
* [6] Lyons, Michael J., Michael Haehnel, and Nobuji Tetsutani. "The mouthesizer: a facial gesture musical interface." Conference Abstracts, Siggraph 2001. Vol. 230. 2001.
* [7] Krägeloh-Mann, Ingeborg, and Christine Cans. "Cerebral palsy update." Brain and development 31.7 (2009): 537-544.
* [8] Sagonas, Christos, et al. "A semi-automatic methodology for facial landmark annotation." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2013.
