use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let Ok(output) = Command::new("clang").arg("--print-runtime-dir").output() else {
        return;
    };
    let directory = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !directory.is_empty()
        && std::path::Path::new(&directory)
            .join("libclang_rt.osx.a")
            .is_file()
    {
        println!("cargo:rustc-link-search=native={directory}");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
