{
  description = "DeepFilterNet3_ll noise suppression as a PipeWire virtual microphone, running on the Intel NPU";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    lib = nixpkgs.lib;
    systems = ["x86_64-linux"];
    eachSystem = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

    # Tooling needs onnx (graph surgery) and OpenVINO (conversion + NPU runtime).
    mkPython = pkgs:
      pkgs.python3.withPackages (ps: [
        ps.numpy
        ps.onnx
        ps.openvino
        ps.soundfile
        ps.scipy
      ]);
  in {
    packages = eachSystem (pkgs: rec {
      python = mkPython pkgs;
      upstreamModel = pkgs.fetchurl {
        url = "https://github.com/Rikorose/DeepFilterNet/raw/v0.5.6/models/DeepFilterNet3_ll_onnx.tar.gz";
        sha256 = "0fnga4r1wd1g4ahx90pj3rbml1ps9aw9gvw6d6vrpq50if7fb62r";
      };
      # The converted, stateful, static-shape NPU artifacts. Everything else
      # (engine, GUI) consumes this; nothing at runtime touches ONNX.
      models = pkgs.callPackage ./nix/models.nix {inherit python upstreamModel;};
      dfmic = pkgs.callPackage ./nix/package.nix {dfmic-models = models;};
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.dfmic;
    });

    devShells = eachSystem (pkgs: {
      default = pkgs.mkShell {
        packages = [
          (mkPython pkgs)
          pkgs.rustc
          pkgs.cargo
          pkgs.rustfmt
          pkgs.clippy
          pkgs.ffmpeg
          pkgs.ladspa-sdk # applyplugin: reference rendering via upstream DeepFilterNet
          pkgs.pkg-config
          # pipewire-sys and libspa-sys run bindgen, which needs libclang.
          pkgs.rustPlatform.bindgenHook
          pkgs.pipewire # dev headers for pipewire-rs
          pkgs.fontconfig
          pkgs.freetype
          pkgs.libGL
          pkgs.wayland # libwayland-client for Slint's winit backend
          pkgs.libxkbcommon
          pkgs.jq
        ];
        shellHook = ''
          export DFMIC_MODELS=${self.packages.${pkgs.stdenv.hostPlatform.system}.models}
          # Hand-written OpenVINO FFI, and pipewire-sys links libpipewire
          # itself: both need the library directories on the native search path.
          export RUSTFLAGS="-L native=${pkgs.pipewire.dev}/lib -L native=${pkgs.openvino.lib}/lib"
          # Slint's winit backend dlopens these, so nothing records them as a
          # dependency; without them `cargo run --features gui` dies with
          # "No default Slint platform was selected".
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.wayland pkgs.libxkbcommon pkgs.libGL]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          echo "dfmic: DFMIC_MODELS is exported, pointing at $DFMIC_MODELS"
        '';
      };
    });

    checks = eachSystem (pkgs: {
      # The DSP tests are pure Rust (no model files), so the package's own
      # `doCheck` compiles and runs them. This check is what makes a broken tree
      # fail `nix flake check`: it builds the whole package, GUI included, and
      # its tests with it. It was missing, so a tree with untracked modules
      # passed the check while `nix build .#dfmic` failed.
      package = self.packages.${pkgs.stdenv.hostPlatform.system}.dfmic;

      # Proves the stateful conversion reproduces the upstream stateless graphs.
      # Fails the build if it drifts.
      #
      # CPU, not NPU: it compares against an FP32 CPU reference, so it has to run
      # on the same kind of device and with the same precision, and it has to
      # work on a machine with no NPU at all.
      parity = let
        python = mkPython pkgs;
      in
        pkgs.runCommand "dfmic-parity" {
          nativeBuildInputs = [python];
        } ''
          cp -r ${./tools/dfnpy} dfnpy
          python dfnpy/verify_convert.py \
            --model-tar ${self.packages.${pkgs.stdenv.hostPlatform.system}.upstreamModel} \
            --ir ${self.packages.${pkgs.stdenv.hostPlatform.system}.models} \
            --device CPU --frames 60 --settle 10 | tee $out
        '';
    });

    nixosModules.default = {pkgs, ...}: {
      imports = [./nix/module.nix];
      services.dfmic.package = lib.mkDefault self.packages.${pkgs.stdenv.hostPlatform.system}.dfmic;
    };

    formatter = eachSystem (pkgs: pkgs.alejandra);
  };
}
