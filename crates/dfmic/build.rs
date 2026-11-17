//! Build the PipeWire bridge.
//!
//! The C file needs PipeWire's headers and the spa pod/buffer layouts, so it is
//! compiled here with `cc` (found via `pkg-config`) rather than through a
//! dependency: the whole bridge is one file and one entry point.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn pkg_config(flag: &str) -> String {
    let out = Command::new("pkg-config")
        .arg(flag)
        .arg("libpipewire-0.3")
        .output()
        .expect("pkg-config is required to build the PipeWire bridge");
    assert!(out.status.success(), "pkg-config failed for libpipewire-0.3");
    String::from_utf8(out.stdout).expect("pkg-config output").trim().to_string()
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let cflags = pkg_config("--cflags");
    let libs = pkg_config("--libs");

    let obj = out_dir.join("pw_bridge.o");
    let lib = out_dir.join("libdfmic_pw.a");

    let mut cc = Command::new("cc");
    cc.arg("-O2").arg("-fPIC").arg("-Wall").arg("-c").arg("c/pw_bridge.c").arg("-o").arg(&obj);
    for flag in cflags.split_whitespace() {
        cc.arg(flag);
    }
    let status = cc.status().expect("failed to run cc");
    assert!(status.success(), "compiling c/pw_bridge.c failed");

    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .arg(&obj)
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
    println!("cargo:rerun-if-changed=build.rs");
}
