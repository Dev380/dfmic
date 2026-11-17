{
  lib,
  rustPlatform,
  pkg-config,
  pipewire,
  openvino,
  stdenv,
  dfmic-models,
  dbus,
  libGL,
  wayland,
  libxkbcommon,
  fontconfig,
  freetype,
  withGui ? true,
}:

# The daemon. The model IR is a separate derivation so the (slow) conversion is
# cached independently of the code.
rustPlatform.buildRustPackage {
  pname = "dfmic";
  version = "0.1.0";

  # The whole workspace: slint-build compiles crates/dfmic/ui/app.slint, and the
  # C bridges live in crates/dfmic/c.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../crates
    ];
  };
  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [pkg-config];
  # dbus is needed by both builds: the control interface is not optional.
  buildInputs = [pipewire openvino dbus]
    ++ lib.optionals withGui [libGL wayland libxkbcommon fontconfig freetype];

  # The OpenVINO C API and PipeWire are found through these search paths; the
  # build script compiles c/pw_bridge.c against PipeWire's headers.
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
    ''
    + lib.optionalString withGui ''
      install -Dm644 ${./dfmic.desktop} $out/share/applications/dfmic.desktop
      for size in 128 256; do
        install -Dm644 ${./icon.png} $out/share/icons/hicolor/''${size}x''${size}/apps/dfmic.png
      done
    '';

  doCheck = false; # the DSP tests run in `nix flake check`, which has the models

  meta = {
    description = "DeepFilterNet3_ll noise suppression as a PipeWire virtual microphone on the Intel NPU";
    license = lib.licenses.mit;
    mainProgram = "dfmic";
    platforms = ["x86_64-linux"];
  };
}
