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
  withGui ? true,
}:
# The daemon. The model IR is a separate derivation so the (slow) conversion is
# cached independently of the code.
rustPlatform.buildRustPackage {
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
      Requires=pipewire.service
      PartOf=pipewire.service

      [Service]
      ExecStart=/run/current-system/sw/bin/dfmic run
      Environment=DFMIC_MODELS=$out/share/dfmic/models
      Restart=on-failure
      RestartSec=5
      Nice=-10

      [Install]
      WantedBy=default.target
      EOF
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
