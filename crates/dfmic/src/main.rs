//! `dfmic` — DeepFilterNet3_ll noise suppression on the Intel NPU.

mod agc;
mod cli;
mod control;
mod daemon;
mod enhance;
#[cfg(feature = "gui")]
mod gui;
mod models;
mod modelwatch;
mod notifications;
mod proto;
mod pw;
mod replay;
mod service;
mod settings;
mod wav;

use std::path::PathBuf;
use std::process::ExitCode;

use dfn_dsp::{MaskReduce, Params};

use crate::proto::Param;

const USAGE: &str = "\
dfmic — DeepFilterNet3_ll virtual microphone on the Intel NPU

usage:
  dfmic run     [--models DIR] [--device NPU] [--source NAME]
                [--default-source auto|never] [--PARAMETER VALUE ...]
  dfmic status  [--json]
  dfmic toggle | on | off
  dfmic set <key> <value>       live parameter (mask-reduce accepts names)
  dfmic reset [--all]
  dfmic source [list | default | <node name>]
  dfmic models [download]       where the model is; `download` fetches it
  dfmic soft on|off
  dfmic agc on|off
  dfmic graph-quantum on|off   lower latency while filtering, at a little CPU
  dfmic monitor on|off
  dfmic default on|off
  dfmic autostart on|off
  dfmic notify on|off
  dfmic dismiss-notices | clear-failure
  dfmic service start

  dfmic doctor  [--models DIR] [--device NPU]
  dfmic enhance <in.wav> <out.wav> [--models DIR] [--device NPU]
                [--atten-lim DB] [--post-filter BETA]
                [--min-thresh DB] [--max-erb-thresh DB] [--max-df-thresh DB]
                [--min-buffer HOPS] [--mask-reduce none|max|mean]
                [--agc] [--agc-trace]
  dfmic gui               the window (needs the gui feature)

parameter keys:
  atten-lim post-filter min-thresh max-erb-thresh max-df-thresh
  min-buffer mask-reduce

environment:
  --models DIR   directory holding enc.xml, erb_dec.xml, df_dec.xml
  DFMIC_MODELS   the same, from the environment
  (with neither, the model is downloaded once into
   $XDG_DATA_HOME/dfmic/models/DeepFilterNet3_ll-0.5.6)
";

struct Args {
    command: String,
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

fn parse_args() -> Result<Args, String> {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().ok_or("no command given")?;
    let mut positional = Vec::new();
    let mut flags = Vec::new();
    while let Some(arg) = argv.next() {
        if let Some(name) = arg.strip_prefix("--") {
            if matches!(name, "help" | "json" | "all" | "agc" | "agc-trace") {
                flags.push((name.to_string(), String::new()));
            } else {
                let value = argv
                    .next()
                    .ok_or_else(|| format!("--{name} needs a value"))?;
                flags.push((name.to_string(), value));
            }
        } else {
            positional.push(arg);
        }
    }
    Ok(Args {
        command,
        positional,
        flags,
    })
}

impl Args {
    fn flag(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(key, _)| key == name)
    }

    /// Where the model is: `--models`, then `DFMIC_MODELS`, then the data
    /// directory. Nothing is downloaded here; the error says how to get it.
    fn models(&self) -> Result<PathBuf, String> {
        models::resolve(self.flag("models").map(PathBuf::from).as_deref())
    }

    fn device(&self) -> String {
        self.flag("device").unwrap_or("NPU").to_string()
    }

    fn number(&self, name: &str) -> Result<Option<f32>, String> {
        match self.flag(name) {
            None => Ok(None),
            Some(value) => value
                .parse::<f32>()
                .map(Some)
                .map_err(|_| format!("--{name} expects a number, got {value:?}")),
        }
    }
}

fn mask_reduce(value: Option<&str>) -> Result<MaskReduce, String> {
    match value.unwrap_or("none") {
        "none" => Ok(MaskReduce::None),
        "max" => Ok(MaskReduce::Max),
        "mean" => Ok(MaskReduce::Mean),
        other => Err(format!(
            "--mask-reduce expects none, max or mean; got {other:?}"
        )),
    }
}

fn params_from(args: &Args) -> Result<Params, String> {
    let mut params = Params::default();
    if let Some(value) = args.number("atten-lim")? {
        params.atten_lim_db = value;
    }
    if let Some(value) = args.number("post-filter")? {
        params.post_filter_beta = value;
    }
    if let Some(value) = args.number("min-thresh")? {
        params.min_db_thresh = value;
    }
    if let Some(value) = args.number("max-erb-thresh")? {
        params.max_db_erb_thresh = value;
    }
    if let Some(value) = args.number("max-df-thresh")? {
        params.max_db_df_thresh = value;
    }
    if let Some(value) = args.number("min-buffer")? {
        if value < 0.0 || value.fract() != 0.0 {
            return Err("--min-buffer expects a non-negative whole number".into());
        }
        params.min_buffer = value as u32;
    }
    params.mask_reduce = mask_reduce(args.flag("mask-reduce"))?;
    Ok(params)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("dfmic: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if args.command == "help" || args.has("help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if matches!(args.command.as_str(), "doctor" | "enhance" | "run") && args.device() != "NPU" {
        eprintln!("dfmic: error: only the NPU is supported; there is deliberately no CPU path");
        return ExitCode::FAILURE;
    }

    let result = dispatch(&args);

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("dfmic: error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Run the command in `args`, after rejecting flags it does not take.
///
/// Split from `main` so every arm is checked in one place: `parse_args` is
/// deliberately permissive about what a flag *is*, and only the command knows
/// which ones it reads.
fn dispatch(args: &Args) -> Result<(), String> {
    let allowed: &[&str] = match args.command.as_str() {
        // The daemon takes the parameter keys as flags, so they are all valid.
        "run" => &[
            "models",
            "device",
            "source",
            "default-source",
            "atten-lim",
            "post-filter",
            "min-thresh",
            "max-erb-thresh",
            "max-df-thresh",
            "min-buffer",
            "mask-reduce",
        ],
        "doctor" => &["models", "device"],
        "enhance" => &[
            "models",
            "device",
            "atten-lim",
            "post-filter",
            "min-thresh",
            "max-erb-thresh",
            "max-df-thresh",
            "min-buffer",
            "mask-reduce",
            "agc",
            "agc-trace",
        ],
        "models" => &["models"],
        "status" => &["json"],
        "reset" => &["all"],
        "set" => &[
            "atten-lim",
            "post-filter",
            "min-thresh",
            "max-erb-thresh",
            "max-df-thresh",
            "min-buffer",
            "mask-reduce",
        ],
        "dump-features" => &["models"],
        _ => &[],
    };
    // `--help` was handled before dispatch; everything else must be consumed.
    let extra: Vec<&str> = args
        .flags
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| *name != "help" && !allowed.contains(name))
        .collect();
    if let Some(name) = extra.first() {
        return Err(format!(
            "{} does not take --{name}\n\n{USAGE}",
            args.command
        ));
    }
    match args.command.as_str() {
        "doctor" => args
            .models()
            .and_then(|models| enhance::doctor(&models, &args.device())),
        "enhance" => run_enhance(args),
        "run" => run_daemon(args),
        "status" => cli::status(args.has("json")),
        "toggle" => cli::toggle(),
        "on" => cli::set_enabled(true),
        "off" => cli::set_enabled(false),
        "set" => match args.positional.as_slice() {
            [key, value] => cli::set_param(key, value),
            _ => Err("set needs <key> <value>".into()),
        },
        "reset" => cli::reset(args.has("all")),
        "models" => match args.positional.as_slice() {
            [] => cli::models(args.flag("models").map(PathBuf::from).as_deref()),
            [command] if command == "download" => {
                cli::models_download(args.flag("models").map(PathBuf::from).as_deref())
            }
            _ => Err("models takes no argument, or `download`".into()),
        },
        "source" if args.positional.len() <= 1 => {
            cli::source(args.positional.first().map(String::as_str))
        }
        "source" => Err("source accepts list, default, or one PipeWire node name".into()),
        "soft" | "agc" | "graph-quantum" | "monitor" | "default" | "autostart" | "notify" => {
            match args.positional.as_slice() {
                [value] => cli::setting(&args.command, value),
                _ => Err(format!("{} needs on or off", args.command)),
            }
        }
        "dismiss-notices" => cli::dismiss_notices(),
        "clear-failure" => cli::clear_failure(),
        "service" if args.positional.as_slice() == ["start"] => cli::service_start(),
        "service" => Err("service supports only `dfmic service start`".into()),
        "gui" => run_gui(),
        "dump-features" => run_dump_features(args),
        "replay-dsp" => {
            if args.positional.len() != 3 {
                Err("replay-dsp needs <trace> <in.wav> <out.wav>".into())
            } else {
                replay::run(
                    std::path::Path::new(&args.positional[0]),
                    std::path::Path::new(&args.positional[1]),
                    std::path::Path::new(&args.positional[2]),
                )
            }
        }
        other => Err(format!("unknown command {other:?}")),
    }
}

#[cfg(feature = "gui")]
fn run_gui() -> Result<(), String> {
    gui::run()
}

#[cfg(not(feature = "gui"))]
fn run_gui() -> Result<(), String> {
    Err("this build has no GUI (build with --features gui)".into())
}

fn run_dump_features(args: &Args) -> Result<(), String> {
    let path = args
        .positional
        .first()
        .ok_or("dump-features needs a wav path")?;
    let models = args.models().ok();
    enhance::dump_features(std::path::Path::new(path), models.as_deref())
}

fn run_daemon(args: &Args) -> Result<(), String> {
    let take_default = match args.flag("default-source") {
        None | Some("auto") => None,
        Some("never") => Some(false),
        Some(other) => {
            return Err(format!(
                "--default-source expects auto or never, got {other:?}"
            ))
        }
    };
    let params = params_from(args)?;
    let mut param_overrides = Vec::new();
    for param in Param::ALL {
        if args.has(param.spec().key) {
            let value = match param {
                Param::AttenLim => params.atten_lim_db as f64,
                Param::PostFilter => params.post_filter_beta as f64,
                Param::MinThresh => params.min_db_thresh as f64,
                Param::MaxErbThresh => params.max_db_erb_thresh as f64,
                Param::MaxDfThresh => params.max_db_df_thresh as f64,
                Param::MinBuffer => params.min_buffer as f64,
                Param::MaskReduce => match params.mask_reduce {
                    MaskReduce::None => 0.0,
                    MaskReduce::Max => 1.0,
                    MaskReduce::Mean => 2.0,
                },
            };
            param_overrides.push((param, value));
        }
    }
    daemon::run(&daemon::Options {
        // Deliberately not `models()`: the daemon runs without a model and
        // downloads it on request, so a missing one must not stop startup.
        models: models::path(args.flag("models").map(PathBuf::from).as_deref()),
        device: args.device(),
        source: args.flag("source").map(str::to_string),
        take_default,
        param_overrides,
    })
}

fn run_enhance(args: &Args) -> Result<(), String> {
    if args.positional.len() != 2 {
        return Err("enhance needs an input and an output path".into());
    }
    enhance::run(&enhance::Options {
        input: PathBuf::from(&args.positional[0]),
        output: PathBuf::from(&args.positional[1]),
        models: args.models()?,
        device: args.device(),
        params: params_from(args)?,
        agc: args.has("agc"),
        agc_trace: args.has("agc-trace"),
    })
}
