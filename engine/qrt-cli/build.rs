use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("repo root");
    let c_dir = repo_root.join("native").join("src");
    let c_files = vec![c_dir.join("qrt.c"), c_dir.join("qwen36_baseline.c")];
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    for c_file in &c_files {
        println!("cargo:rerun-if-changed={}", c_file.display());
    }
    println!("cargo:rerun-if-changed={}", c_dir.join("qrt.h").display());
    println!(
        "cargo:rerun-if-changed={}",
        c_dir.join("qwen36_baseline.h").display()
    );

    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build_msvc(&c_dir, &c_files, &out_dir);
    } else {
        build_unix(&c_dir, &c_files, &out_dir);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qrt_c");
}

fn build_msvc(c_dir: &Path, c_files: &[PathBuf], out_dir: &Path) {
    let lib_file = out_dir.join("qrt_c.lib");
    let mut obj_files = Vec::new();

    for c_file in c_files {
        let stem = c_file
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("C file stem");
        let obj_file = out_dir.join(format!("{stem}.obj"));
        let cl_status = Command::new("cl")
            .arg("/nologo")
            .arg("/std:c11")
            .arg("/O2")
            .arg("/W4")
            .arg("/c")
            .arg(format!("/I{}", c_dir.display()))
            .arg(c_file)
            .arg(format!("/Fo{}", obj_file.display()))
            .status()
            .expect("failed to run cl.exe");
        assert!(cl_status.success(), "cl.exe failed");
        obj_files.push(obj_file);
    }

    let mut lib_command = Command::new("lib");
    lib_command
        .arg("/nologo")
        .arg(format!("/OUT:{}", lib_file.display()));
    for obj_file in &obj_files {
        lib_command.arg(obj_file);
    }
    let lib_status = lib_command.status().expect("failed to run lib.exe");
    assert!(lib_status.success(), "lib.exe failed");
}

fn build_unix(c_dir: &Path, c_files: &[PathBuf], out_dir: &Path) {
    let lib_file = out_dir.join("libqrt_c.a");
    let mut obj_files = Vec::new();

    for c_file in c_files {
        let stem = c_file
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("C file stem");
        let obj_file = out_dir.join(format!("{stem}.o"));
        let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let cc_status = Command::new(cc)
            .arg("-std=c11")
            .arg("-O2")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg(format!("-I{}", c_dir.display()))
            .arg("-c")
            .arg(c_file)
            .arg("-o")
            .arg(&obj_file)
            .status()
            .expect("failed to run C compiler");
        assert!(cc_status.success(), "C compiler failed");
        obj_files.push(obj_file);
    }

    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let mut ar_command = Command::new(ar);
    ar_command.arg("crs").arg(&lib_file);
    for obj_file in &obj_files {
        ar_command.arg(obj_file);
    }
    let ar_status = ar_command.status().expect("failed to run ar");
    assert!(ar_status.success(), "ar failed");
}
