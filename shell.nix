with (import <nixpkgs> { });

mkShell {
  buildInputs = [
    (dlib.override { guiSupport = true; })
    gst_all_1.gstreamer
    gtk2 # not necessary
    gtk2.dev # not necessary
    opencv
    opencv2.dev
  ];
  nativeBuildInputs = [
    bzip2
    fribidi.out # not necessary
    libselinux.dev # not necessary
    libsepol.dev # not necessary
    libthai # not necessary
    libdatrie # not necessary
    cmake
    xorg.libXdmcp.dev # not necessary
    util-linux.dev # not necessary
    patchelf
    pkg-config
    pkg-config-unwrapped # not necessary
    pcre2.dev # not necessary
    pcre.dev # not necessary
  ];
  shellHook = ''
    echo '${libthai}'
    echo '${libthai.out}'
    if [[ -n $LD_LIBRARY_PATH ]]; then
      LD_LIBRARY_PATH="${opencv.out}/lib:$LD_LIBRARY_PATH"
    else
      LD_LIBRARY_PATH="${opencv.out}/lib"
    fi
  '';
}
