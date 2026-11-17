//! Build the PipeWire bridge.
//!
//! The C file needs PipeWire's headers and the spa pod/buffer layouts, so it is
//! compiled here with `cc` (found via `pkg-config`) rather than through a
//! dependency: the whole bridge is one file and one entry point.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Run pkg-config for our two C dependencies and return its output.

fn pkg_config(args: &[&str]) -> String {
    let out = Command::new("pkg-config")
        .args(args)
        .output()
        .expect("pkg-config is required to build the C bridges");
    assert!(
        out.status.success(),
        "pkg-config {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("pkg-config output").trim().to_string()
}

fn main() {
    // The GUI is optional; when it is on, compile the .slint source to Rust here
    // so the crate needs no slint-build dependency of its own.
    let gui = std::env::var_os("CARGO_FEATURE_GUI").is_some();
    println!("cargo:rerun-if-changed=ui/app.slint");
    if gui {
        #[cfg(feature = "gui")]
        slint_build::compile("ui/app.slint").expect("compiling ui/app.slint");
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    // -l/-L are link-time only; keep the include search paths for `cc -c`.
    let cflags: String = pkg_config(&["--cflags", "libpipewire-0.3", "dbus-1"]);
    let libs = pkg_config(&["--libs", "libpipewire-0.3", "dbus-1"]);

    let obj = out_dir.join("pw_bridge.o");
    let dbus_obj = out_dir.join("dbus_ctl.o");
    let lib = out_dir.join("libdfmic_pw.a");

    let mut cc = Command::new("cc");
    cc.arg("-O2").arg("-fPIC").arg("-Wall").arg("-c").arg("c/pw_bridge.c").arg("-o").arg(&obj);
    for flag in cflags.split_whitespace() {
        cc.arg(flag);
    }
    let status = cc.status().expect("failed to run cc");
    assert!(status.success(), "compiling c/pw_bridge.c failed");

    let mut dbus_cc = Command::new("cc");
    dbus_cc
        .arg("-O2")
        .arg("-fPIC")
        .arg("-Wall")
        .arg("-c")
        .arg("c/dbus_ctl.c")
        .arg("-o")
        .arg(&dbus_obj);
    for flag in cflags.split_whitespace() {
        // -l/-L flags are link-time only; keep the search paths so `cc -c` sees
        // headers but not libraries.
        if flag.starts_with("-l") {
            continue;
        }
        if flag.starts_with("-L") {
            dbus_cc.arg("-Wl,--start-group").arg(flag).arg("-Wl,--end-group");
            continue;
        }
        dbus_cc.arg(flag);
    }
    let status = dbus_cc.status().expect("failed to run cc for dbus_ctl.c");
    assert!(status.success(), "compiling c/dbus_ctl.c failed");

    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .arg(&obj)
        .arg(&dbus_obj)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "archiving libdfmic_pw.a failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=dfmic_pw");
    for flag in libs.split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(name) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={name}");
        }
    }
    println!("cargo:rerun-if-changed=c/pw_bridge.c");
    println!("cargo:rerun-if-changed=c/dbus_ctl.c");
    println!("cargo:rerun-if-changed=build.rs");
}
