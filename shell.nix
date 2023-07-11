with (import <nixpkgs> { });

mkShell {
  buildInputs = [
    (dlib.override { guiSupport = true; })
    gst_all_1.gstreamer
    gtk2
    gtk2.dev
    opencv
    opencv2.dev
    pkg-config-unwrapped
  ];
  nativeBuildInputs = [
    bzip2
    cmake
    patchelf
    pkg-config
  ];
  shellHook = ''
    echo '${pkg-config-unwrapped}'
    if [[ -n $LD_LIBRARY_PATH ]]; then
      LD_LIBRARY_PATH="${opencv.out}/lib:$LD_LIBRARY_PATH"
    else
      LD_LIBRARY_PATH="${opencv.out}/lib"
    fi
  '';
}
