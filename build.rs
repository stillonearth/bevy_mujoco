use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

fn get_output_path() -> PathBuf {
    let manifest_dir_string = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_type = env::var("PROFILE").unwrap();
    let path = Path::new(&manifest_dir_string)
        .join("target")
        .join(build_type);
    path
}

fn main() {
    // This build script is mainly for Windows DLL copying
    // The mujoco-rs crate handles the actual linking via MUJOCO_DYNAMIC_LINK_DIR
    
    let (_, _, default_install) = match env::var("CARGO_CFG_UNIX") {
        Ok(_) => ("", "", ""),
        _ => match env::var("CARGO_CFG_WINDOWS") {
            Ok(_) => ("", "dll", "C:\\Program Files\\MuJoCo"),
            _ => ("", "", ""),
        },
    };

    if option_env!("DOCS_RS").is_none() {
        let mj_root = match (env::var("MUJOCO_DIR"), env::var("MUJOCO_PREFIX")) {
            (Ok(dir), _) | (Err(..), Ok(dir)) => dir,
            (Err(..), Err(..)) => default_install.to_string(),
        };
        
        // Default to ~/.mujoco/mujoco-3.3.7 if not set
        let mj_root = if mj_root.is_empty() {
            format!("{}/.mujoco/mujoco-3.3.7", env::var("HOME").unwrap_or_default())
        } else {
            mj_root
        };
        
        let mj_root = PathBuf::from_str(&mj_root).expect("Unable to get path");
        let mj_lib_windows = mj_root.join("bin");

        // Copy mujoco.dll to target directory on Windows targets
        if env::var("CARGO_CFG_WINDOWS").is_ok() {
            let target_dir = get_output_path();
            let src = Path::join(
                &env::current_dir().unwrap(),
                mj_lib_windows.join("mujoco.dll"),
            );

            fs::create_dir_all(&target_dir).unwrap();
            let dest = Path::join(Path::new(&target_dir), Path::new("mujoco.dll"));
            eprintln!("Copying {src:?} to {dest:?}");
            std::fs::copy(src, dest).unwrap();
        }

        // For Unix systems, link MuJoCo library
        if env::var("CARGO_CFG_UNIX").is_ok() {
            println!("cargo:rustc-link-search={}/lib", mj_root.display());
            println!("cargo:rustc-link-lib=mujoco");
        }
    }
}
