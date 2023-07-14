with (import <nixpkgs> { });

let
  libremidi = stdenv.mkDerivation {
    name = "libremidi";
    version = "3.0";
    src = fetchgit {
      url = "https://github.com/jcelerier/libremidi";
      rev = "v3.0";
      sha256 = "aO83a0DmzwjYXDlPIsn136EkDF0406HadTXPoGuVF6I="; # https://dev.to/deciduously/workstation-management-with-nix-flakes-build-a-cmake-c-package-21lp
    };
    nativeBuildInputs = [
      alsa-lib
      gcc
      cmake
    ];
  };
in
mkShell {
  buildInputs = [
    (dlib.override { guiSupport = true; })
    gst_all_1.gstreamer
    gtk2 # not necessary
    gtk2.dev # not necessary
    opencv
    opencv2.dev
    libremidi
    alsa-lib
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
    echo '${alsa-lib}'
    echo '${libremidi}'
    if [[ -n $LD_LIBRARY_PATH ]]; then
      LD_LIBRARY_PATH="${opencv.out}/lib:$LD_LIBRARY_PATH"
    else
      LD_LIBRARY_PATH="${opencv.out}/lib"
    fi
  '';
}
