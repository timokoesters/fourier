fn main() {
    println!("cargo:rerun-if-changed=.");
    if std::env::var("TARGET").unwrap() != "wasm32-unknown-unknown" {
        let success = std::process::Command::new("wasm-pack")
            .args(&["build", "-t", "web"])
            .spawn()
            .unwrap()
            .wait()
            .unwrap()
            .success();
        assert!(success);
    }
}
