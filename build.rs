use std::env;

fn main() {
    let whisper_dir = "third_party/whisper.cpp";

    let mut cmake_cfg = cmake::Config::new(whisper_dir);

    cmake_cfg
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    // GPU acceleration
    if cfg!(feature = "cuda") {
        cmake_cfg.define("GGML_CUDA", "ON");
    }

    if cfg!(feature = "metal") || (cfg!(target_os = "macos") && !cfg!(feature = "cuda")) {
        // Enable Metal by default on macOS unless CUDA is explicitly requested
        cmake_cfg.define("GGML_METAL", "ON");
    }

    let dst = cmake_cfg.build();

    // Link the built static library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // Link whisper and ggml
    println!("cargo:rustc-link-lib=static=whisper");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    // Platform-specific linking
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=static=ggml-blas");
        if cfg!(feature = "metal") || !cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=static=ggml-metal");
        }
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=gomp");
        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=static=ggml-cuda");
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=cudart");
        }
    }

    // Rebuild if whisper.cpp sources change
    println!("cargo:rerun-if-changed={}/src", whisper_dir);
    println!("cargo:rerun-if-changed={}/include", whisper_dir);
    println!("cargo:rerun-if-changed={}/ggml", whisper_dir);
    println!("cargo:rerun-if-changed=build.rs");
}
