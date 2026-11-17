{
  description = "DeepFilterNet3_ll noise suppression as a PipeWire virtual microphone, running on the Intel NPU";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
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
    packages = eachSystem (pkgs: let
      python = mkPython pkgs;
      upstreamModel = pkgs.fetchurl {
        url = "https://github.com/Rikorose/DeepFilterNet/raw/v0.5.6/models/DeepFilterNet3_ll_onnx.tar.gz";
        sha256 = "0fnga4r1wd1g4ahx90pj3rbml1ps9aw9gvw6d6vrpq50if7fb62r";
      };
    in {
      # The converted, stateful, static-shape NPU artifacts. Everything else
      # (engine, GUI) consumes this; nothing at runtime touches ONNX.
      models = pkgs.callPackage ./nix/models.nix {inherit python upstreamModel;};
      inherit python upstreamModel;
      default = self.packages.${pkgs.system}.models;
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
          pkgs.jq
        ];
        shellHook = ''
          export DFMIC_MODELS=${self.packages.${pkgs.system}.models}
          echo "dfmic: model artifacts at $DFMIC_MODELS"
        '';
      };
    });

    checks = eachSystem (pkgs: {
      # Proves the stateful conversion reproduces the upstream stateless graphs.
      # Fails the build if it drifts. CPU here so it also runs without an NPU.
      parity = let
        python = mkPython pkgs;
      in
        pkgs.runCommand "dfmic-parity" {
          nativeBuildInputs = [python];
        } ''
          cp -r ${./tools/dfnpy} dfnpy
          python dfnpy/verify_convert.py \
            --model-tar ${self.packages.${pkgs.system}.upstreamModel} \
            --ir ${self.packages.${pkgs.system}.models} \
            --device CPU --frames 60 --settle 10 | tee $out
        '';
    });

    formatter = eachSystem (pkgs: pkgs.alejandra);
  };
}
