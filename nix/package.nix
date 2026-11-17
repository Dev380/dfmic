{
  lib,
  rustPlatform,
  pkg-config,
  pipewire,
  openvino,
  stdenv,
  dfmic-models,
}:

# The daemon. The model IR is a separate derivation so the (slow) conversion is
# cached independently of the code.
rustPlatform.buildRustPackage {
  pname = "dfmic";
  version = "0.1.0";

  src = ../.;
  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [pkg-config];
  buildInputs = [pipewire openvino];

  # The OpenVINO C API and PipeWire are found through these search paths; the
  # build script compiles c/pw_bridge.c against PipeWire's headers.
  env.RUSTFLAGS = "-L native=${pipewire.dev}/lib -L native=${openvino.lib}/lib";

  postInstall = ''
    mkdir -p $out/share/dfmic
    ln -s ${dfmic-models} $out/share/dfmic/models
  '';

  doCheck = false; # the DSP tests run in `nix flake check`, which has the models

  meta = {
    description = "DeepFilterNet3_ll noise suppression as a PipeWire virtual microphone on the Intel NPU";
    license = lib.licenses.mit;
    mainProgram = "dfmic";
    platforms = ["x86_64-linux"];
  };
}
