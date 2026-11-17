{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.dfmic;
  # The package is passed in by the flake so the module has no fixed nixpkgs.
  package = cfg.package;
in {
  options.services.dfmic = {
    enable = lib.mkEnableOption "DeepFilterNet3_ll virtual microphone on the Intel NPU";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The dfmic package to run.";
    };

    device = lib.mkOption {
      type = lib.types.str;
      default = "NPU";
      description = "OpenVINO device. There is deliberately no CPU fallback.";
    };

    source = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Physical source to capture from (node name). `null` follows the system
        default, which must be a real device while dfmic is starting up.
      '';
    };

    gui = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Add the Slint control panel to the user's environment and a desktop
        entry for it. The daemon itself is headless either way.
      '';
    };

    takeDefault = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Make the virtual microphone the default source while running.";
    };
  };

  config = lib.mkIf cfg.enable {
    # The window is a client of the daemon, so it lives in the user profile
    # rather than in the service.
    home.packages = lib.mkIf cfg.gui [package];
    # The daemon talks to PipeWire and to the NPU.
    users.groups.render = {};
    systemd.user.services.dfmic = {
      description = "DeepFilterNet3_ll virtual microphone";
      partOf = ["pipewire.service"];
      after = ["pipewire.service" "wireplumber.service"];
      requires = ["pipewire.service"];
      serviceConfig = {
        ExecStart = lib.concatStringsSep " " (
          ["${package}/bin/dfmic run" "--device" cfg.device]
          ++ lib.optional (cfg.source != null) "--source ${cfg.source}"
          ++ lib.optional (!cfg.takeDefault) "--default-source never"
        );
        Environment = "DFMIC_MODELS=${package}/share/dfmic/models";
        # Fail loudly and let systemd retry with a backoff; the daemon restores
        # the previous default source before exiting.
        Restart = "on-failure";
        RestartSec = 5;
        # Audio work is realtime-sensitive.
        Nice = -10;
      };
      wantedBy = ["default.target"];
    };
  };
}
