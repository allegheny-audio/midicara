with (import <nixpkgs> { });

mkShell {
  buildInputs = [
    (dlib.override { guiSupport = true; })
    opencv
    opencv2.dev
    gst_all_1.gstreamer
  ];
  nativeBuildInputs = [
    cmake
    pkg-config
    patchelf
    bzip2
  ];
  shellHook = ''
    echo '${opencv2.dev}'
    if [[ -n $LD_LIBRARY_PATH ]]; then
      LD_LIBRARY_PATH="${opencv.out}/lib:$LD_LIBRARY_PATH"
    else
      LD_LIBRARY_PATH="${opencv.out}/lib"
    fi
  '';
}
