{
  pkgs,
  python,
  upstreamModel,
}:
# Converts the upstream DeepFilterNet3_ll ONNX graphs into stateful, static-shape
# OpenVINO IR that the NPU plugin can compile. See tools/dfnpy/convert.py for why
# the graphs need rewriting at all.
pkgs.stdenvNoCC.mkDerivation {
  pname = "dfmic-models";
  version = "DeepFilterNet3_ll-0.5.6";

  dontUnpack = true;
  nativeBuildInputs = [python];

  buildPhase = ''
    runHook preBuild
    python ${../tools/dfnpy/convert.py} \
      --model-tar ${upstreamModel} \
      --out $out
    runHook postBuild
  '';

  dontInstall = true;

  meta = {
    description = "DeepFilterNet3_ll converted to stateful OpenVINO IR for the Intel NPU";
    # Upstream DeepFilterNet is dual-licensed; the converted IR inherits it.
    license = with pkgs.lib.licenses; OR [mit asl20];
  };
}
