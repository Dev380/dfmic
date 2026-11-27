{
  config,
  lib,
  pkgs,
  utils,
  ...
}: let
  cfg = config.services.dfmic;
  package = cfg.package;

  # Everything the daemon forks: `pw-metadata` (default-source takeover, its
  # startup recovery and the metadata fallback), curl + sha256sum + tar + gzip
  # (the model downloader), and `systemctl` (unit queries from the GUI).
  #
  # NixOS user units otherwise get its minimal default PATH, which holds only
  # coreutils/findutils/gnugrep/gnused/systemd. Without `pw-metadata` the daemon
  # cannot restore the default source after an unclean stop and refuses to
  # start, which `Restart=on-failure` then turns into a restart loop.
  # `tar -xzf` shells out to `gzip`, so gzip has to be on PATH too.
  servicePath = with pkgs; [pipewire curl coreutils gnutar gzip systemd];
in {
  options.services.dfmic = {
    enable = lib.mkEnableOption "DeepFilterNet3_ll virtual microphone on the Intel NPU";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The dfmic package to install and run.";
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
        Physical source to capture from (node name). `null` selects the current
        system default on first run and persists that choice.
      '';
    };

    takeDefault = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Make the virtual microphone the default source while filtering is available.";
    };

    realtime = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Let the NPU worker ask RealtimeKit for realtime scheduling (below
        PipeWire's own audio threads).

        This shortens the worker's worst case, which is what decides whether the
        output ring starves under load. It does not lower latency: the buffering
        floor comes from bursty hop production, not from jitter.

        Needs `security.rtkit.enable`; without it the daemon says so once and
        runs at normal priority, which still works.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # This is a NixOS module throughout: no home-manager options are required.
    # systemd.packages imports lib/systemd/user/dfmic.service, including its
    # [Install] section. Enabling it remains a per-user decision.
    environment.systemPackages = [package];
    systemd.packages = [package];

    # Realtime scheduling goes through RealtimeKit, which answers "yes" for an
    # active local session without a prompt. The alternatives do not work: a
    # capability wrapper makes `CAP_SYS_NICE` ambient in every process that
    # touches the binary, and a user unit's `LimitRTPRIO=` is accepted by systemd
    # and then ignored, because the user manager itself runs at 0.
    security.rtkit.enable = lib.mkIf cfg.realtime true;

    # Keep the package's upstream user unit and Install section. This drop-in
    # overrides the argv and PATH, and adds the module-only settings.
    #
    # `path` is the declarative way to set a unit's PATH; `utils.escapeSystemdExecArgs`
    # quotes every argv element, so a source name with spaces ("Built-in Audio
    # Analog Stereo") stays one argument instead of becoming a capture target
    # plus four stray parameters.
    #
    # (`utils` is the NixOS special arg; `lib` has no escapeSystemdExecArgs.)
    systemd.user.services.dfmic = {
      overrideStrategy = "asDropin";
      path = servicePath;
      startLimitBurst = 5;
      startLimitIntervalSec = 60;
      serviceConfig.ExecStart = [
        ""
        (utils.escapeSystemdExecArgs (
          [
            "${package}/bin/dfmic"
            "run"
            "--device"
            cfg.device
          ]
          ++ lib.optional (cfg.source != null) "--source"
          ++ lib.optional (cfg.source != null) cfg.source
          ++ lib.optional (!cfg.takeDefault) "--default-source"
          ++ lib.optional (!cfg.takeDefault) "never"
        ))
      ];
    };
  };
}
