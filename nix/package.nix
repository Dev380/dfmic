{
  lib,
  rustPlatform,
  pkg-config,
  pipewire,
  openvino,
  stdenv,
  dfmic-models,
  libGL,
  wayland,
  libxkbcommon,
  fontconfig,
  freetype,
  curl,
  coreutils,
  gnutar,
  gzip,
  systemd,
  withGui ? true,
}:
# The daemon. The model IR is a separate derivation so the (slow) conversion is
# cached independently of the code.
rustPlatform.buildRustPackage (let
  # Everything the daemon forks: `pw-metadata` (default-source takeover and its
  # startup recovery), curl + sha256sum + tar + gzip (the model downloader) and
  # `systemctl` (unit queries from the GUI). NixOS user units get its minimal
  # default PATH otherwise, which holds none of these.
  servicePath = lib.makeBinPath [pipewire curl coreutils gnutar gzip systemd];
in {
  pname = "dfmic";
  version = "0.1.0";

  # The whole workspace: slint-build compiles crates/dfmic/ui/main.slint, which
  # embeds the window icon from nix/.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../crates
      ../nix/icon.png
    ];
  };
  cargoLock.lockFile = ../Cargo.lock;

  # pipewire-sys and libspa-sys run bindgen, which needs libclang.
  nativeBuildInputs = [pkg-config rustPlatform.bindgenHook];
  buildInputs =
    [pipewire openvino]
    ++ lib.optionals withGui [libGL wayland libxkbcommon fontconfig freetype];

  # The OpenVINO C API and PipeWire are found through these search paths.
  env.RUSTFLAGS = "-L native=${pipewire.dev}/lib -L native=${openvino.lib}/lib";
  # The GUI is optional: the daemon and the offline path build and run headless.
  cargoBuildFlags = lib.optional withGui "--features gui";

  # Slint's winit backend loads libwayland with dlopen, so the auto-patcher
  # cannot see it; without this the GUI fails with "wayland library could not
  # be loaded" even though the library is in the closure.
  postFixup = lib.optionalString withGui ''
    patchelf --add-rpath ${lib.makeLibraryPath [wayland libxkbcommon libGL]} $out/bin/dfmic
  '';

  postInstall =
    ''
      mkdir -p $out/share/dfmic
      ln -s ${dfmic-models} $out/share/dfmic/models
      mkdir -p $out/lib/systemd/user
      cat > $out/lib/systemd/user/dfmic.service <<EOF
      [Unit]
      Description=DeepFilterNet3_ll virtual microphone on the Intel NPU
      After=pipewire.service wireplumber.service
      Wants=wireplumber.service
      Requires=pipewire.service
      PartOf=pipewire.service

      [Service]
      ExecStart=$out/bin/dfmic run
      Environment=DFMIC_MODELS=$out/share/dfmic/models
      Environment=PATH=${servicePath}
      Restart=on-failure
      RestartSec=5

      [Install]
      WantedBy=default.target
      EOF
    ''
    + lib.optionalString withGui ''
      install -Dm644 ${./dfmic.desktop} $out/share/applications/dfmic.desktop
      # The source PNG is 256x256. Installing it at 128x128 too would declare a
      # size it is not, so desktop shells would scale it anyway.
      install -Dm644 ${./icon.png} $out/share/icons/hicolor/256x256/apps/dfmic.png
    '';

  # The tests are pure Rust and need no model files, so they run here. The test
  # flags do not inherit `cargoBuildFlags`, so the GUI feature has to be named
  # again: without it the four GUI-logic tests are not compiled, and this build
  # would not check the code it just produced.
  doCheck = true;
  cargoTestFlags = lib.optionals withGui ["--features" "gui"];

  meta = {
    description = "DeepFilterNet3_ll noise suppression as a PipeWire virtual microphone on the Intel NPU";
    # Same as the crates' own `license` field (MIT OR Apache-2.0); the dev shell,
    # the daemon and the model IR are all dual-licensed.
    license = lib.licenses.OR [lib.licenses.mit lib.licenses.asl20];
    mainProgram = "dfmic";
    platforms = ["x86_64-linux"];
  };
})
