On windows, go to [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install) and follow the instruction for installing Rust. You, might want to download <span style="color:rgb(250, 128, 114)">visual studio</span>. On <span style="color:rgb(250, 128, 114)">visual studio code</span> we have to install <span style="color:rgb(250, 128, 114)">Rust-analyzer</span> and <span style="color:rgb(250, 128, 114)">Rust</span> extension.
### ==Troubleshooting==
To check whether you have Rust installed correctly, open a shell and enter this line:

```rust
$ rustc --version
```

You should see the version number, commit hash, and commit date for the latest stable version that has been released, in the following format:

```rust
rustc x.y.z (abcabcabc yyyy-mm-dd)
```

###  ==Updating and Uninstalling==
Once Rust is installed via `rustup`, updating to a new released version is east. From your shell, run the following update script:

```rust
$ rustup update
```

To uninstall Rust and `rustup`, run the following uninstall script from your shell:

```rust
$ rustup self uninstall
```

