# Midicara

Midicara is a software midi controller that outputs midi signals based on facial landmarks. These facial landmarks a standard developed by **i-bug** and can be found [here](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).

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
