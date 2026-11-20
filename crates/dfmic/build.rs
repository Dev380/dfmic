//! Compile the Slint sources when the `gui` feature is on.
//!
//! The PipeWire bridge is pure Rust now, so there is no C to build here.

fn main() {
    // The GUI is optional; when it is on, compile the .slint source to Rust here
    // so the crate needs no slint-build dependency of its own.
    let gui = std::env::var_os("CARGO_FEATURE_GUI").is_some();
    println!("cargo:rerun-if-changed=ui");
    if gui {
        #[cfg(feature = "gui")]
        slint_build::compile_with_config(
            "ui/main.slint",
            slint_build::CompilerConfiguration::new().with_style("fluent".into()),
        )
        .expect("compiling ui/main.slint");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
