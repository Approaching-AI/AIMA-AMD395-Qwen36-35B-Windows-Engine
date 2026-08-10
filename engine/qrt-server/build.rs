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
    let c_files = vec![
        c_dir.join("qrt.c"),
        c_dir.join("qwen36_baseline.c"),
        c_dir.join("qrt_server_bridge.c"),
    ];
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    if let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if output.status.success() {
            let sha = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            println!("cargo:rustc-env=QRT_GIT_SHA={sha}");
        }
    }
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join(".git/HEAD").display()
    );

    for c_file in &c_files {
        println!("cargo:rerun-if-changed={}", c_file.display());
    }
    for header in ["qrt.h", "qwen36_baseline.h", "qrt_server_bridge.h"] {
        println!("cargo:rerun-if-changed={}", c_dir.join(header).display());
    }

    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build_msvc(&c_dir, &c_files, &out_dir);
        println!("cargo:rustc-link-arg=/STACK:268435456");
    } else {
        build_unix(&c_dir, &c_files, &out_dir);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qrt_server_c");
    if env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=m");
    }
}

fn build_msvc(c_dir: &Path, c_files: &[PathBuf], out_dir: &Path) {
    let lib_file = out_dir.join("qrt_server_c.lib");
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
            .arg("/D_CRT_SECURE_NO_WARNINGS")
            .arg("/DQRT_STATIC")
            .arg("/c")
            .arg(format!("/I{}", c_dir.display()))
            .arg(c_file)
            .arg(format!("/Fo{}", obj_file.display()))
            .status()
            .expect("failed to run cl.exe; run Cargo from VsDevCmd.bat");
        assert!(
            cl_status.success(),
            "cl.exe failed for {}",
            c_file.display()
        );
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
    let lib_file = out_dir.join("libqrt_server_c.a");
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
            .arg("-DQRT_STATIC")
            .arg(format!("-I{}", c_dir.display()))
            .arg("-c")
            .arg(c_file)
            .arg("-o")
            .arg(&obj_file)
            .status()
            .expect("failed to run C compiler");
        assert!(
            cc_status.success(),
            "C compiler failed for {}",
            c_file.display()
        );
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
