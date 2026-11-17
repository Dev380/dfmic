//! `dfmic` — DeepFilterNet3_ll noise suppression on the Intel NPU.
//!
//! Subcommands:
//!   enhance  clean a file offline (also the parity harness's entry point)
//!   doctor   check the NPU and the model artifacts
//!   run      the daemon: a PipeWire virtual microphone
//!   status   ask a running daemon what it is doing

mod daemon;
mod enhance;
mod pw;
mod replay;
mod wav;

use std::path::PathBuf;
use std::process::ExitCode;

use dfn_dsp::Params;

const USAGE: &str = "\
dfmic — DeepFilterNet3_ll virtual microphone on the Intel NPU

usage:
  dfmic doctor  [--models DIR] [--device NPU]
  dfmic enhance <in.wav> <out.wav> [--models DIR] [--device NPU]
                [--atten-lim DB] [--post-filter BETA]
                [--min-thresh DB] [--max-erb-thresh DB] [--max-df-thresh DB]
  dfmic run     [--models DIR] [--device NPU] [--source NAME]
                [--default-source auto|never]
  dfmic status

environment:
  DFMIC_MODELS   directory holding enc.xml, erb_dec.xml, df_dec.xml
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
            if name == "help" {
                flags.push((name.to_string(), String::new()));
            } else {
                let value = argv.next().ok_or_else(|| format!("--{name} needs a value"))?;
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
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(k, _)| k == name)
    }

    fn models(&self) -> Result<PathBuf, String> {
        if let Some(dir) = self.flag("models") {
            return Ok(PathBuf::from(dir));
        }
        if let Ok(dir) = std::env::var("DFMIC_MODELS") {
            return Ok(PathBuf::from(dir));
        }
        Err("no model directory: pass --models DIR or set DFMIC_MODELS".into())
    }

    fn device(&self) -> String {
        self.flag("device").unwrap_or("NPU").to_string()
    }

    fn number(&self, name: &str) -> Result<Option<f32>, String> {
        match self.flag(name) {
            None => Ok(None),
            Some(v) => v
                .parse::<f32>()
                .map(Some)
                .map_err(|_| format!("--{name} expects a number, got {v:?}")),
        }
    }
}

fn params_from(args: &Args) -> Result<Params, String> {
    let mut params = Params::default();
    if let Some(v) = args.number("atten-lim")? {
        params.atten_lim_db = v;
    }
    if let Some(v) = args.number("post-filter")? {
        params.post_filter_beta = v;
    }
    if let Some(v) = args.number("min-thresh")? {
        params.min_db_thresh = v;
    }
    if let Some(v) = args.number("max-erb-thresh")? {
        params.max_db_erb_thresh = v;
    }
    if let Some(v) = args.number("max-df-thresh")? {
        params.max_db_df_thresh = v;
    }
    Ok(params)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("dfmic: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if args.command == "help" || args.has("help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let result = match args.command.as_str() {
        "doctor" => args
            .models()
            .and_then(|m| enhance::doctor(&m, &args.device())),
        "enhance" => run_enhance(&args),
        "run" => run_daemon(&args),
        "status" => daemon::status(),
        "dump-features" => run_dump_features(&args),
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
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Every failure path here is loud on purpose: a silent fallback would
            // leave the user believing their mic is filtered when it is not.
            eprintln!("dfmic: error: {e}");
            ExitCode::FAILURE
        }
    }
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
    daemon::run(&daemon::Options {
        models: args.models()?,
        device: args.device(),
        source: args.flag("source").map(str::to_string),
        take_default: args.flag("default-source").unwrap_or("auto") == "auto",
        params: params_from(args)?,
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
    })
}
