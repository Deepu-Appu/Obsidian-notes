<span style="color:rgb(250, 128, 114)">Cargo</span> is Rust's build system and package manager. Most Rustaceans use this tool to manage their Rust projects because <span style="color:rgb(250, 128, 114)">Cargo</span> handles a lot of task for you, such as building your code, downloading the libraries your code depends on, and building those libraries. (we call the libraries that your code needs <span style="color:rgb(250, 128, 114)">dependencies</span>.)

The simplest Rust programs, like the one we've written so far, don't have any dependencies. If we had built the `"Hello, world!"` project with<span style="color:rgb(250, 128, 114)"> Cargo</span> , it would only use the part of <span style="color:rgb(250, 128, 114)">Cargo </span>that handles building your code. As you write more complex Rust programs, you'll add dependencies, and if you start a project using <span style="color:rgb(250, 128, 114)">Cargo,</span> adding dependencies will be much easier to do.

To check if <span style="color:rgb(250, 128, 114)">Cargo</span> is installed or not you can check it by:

```powershell
cargo --version
```

### ==Creating a Project with Cargo==

Let's create a new project using <span style="color:rgb(250, 128, 114)">Cargo</span>. Navigate back `project` directory (or wherever you decided to store your code.) Then, on any operating system, run the following:

```powershell
> cargo new hello_cargo
> cd hello_cargo
```

The first command creates a new directory and project called `hello_cargo`. We've named our project `hello_cargo`, and <span style="color:rgb(250, 128, 114)">Cargo</span> creates its files in a directory of the same name. 

Go into the `hello_cargo` directory and list the files. You'll see that <span style="color:rgb(250, 128, 114)">Cargo </span>has generated two files and one directory for us: a `Cargo.toml` file and a src directory with a `main.rs` file inside.

It has also initialized a new Git repository along with a `.gitignore` file. Git files won't be generated if you run `cargo new` within an existing repository; you can override this behavior by using `cargo new --vcs=git`

Open `Cargo.toml` in your  text editor of choice. 

```rust
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2024"

[dependencies]
```

This first is in the `TOML` (Tom's Obvious, Minimal Language) format, which is <span style="color:rgb(250, 128, 114)">cargo's </span>configuration format.

`[package]` -> configuration of Rust the package name, version of the rust and the edition.

`[dependencies]` -> the dependencies we want for our program.

If you started a project that doesn't use <span style="color:rgb(250, 128, 114)">Cargo</span>, as we did with the "Hello, world!" project, you can convert it to a project that does use<span style="color:rgb(250, 128, 114)"> Cargo</span>. Move the project code into the `src` directory and create an appropriate `Cargo.toml` file. One easy way to get that `Cargo.toml` is to run `cargo init`, which will create it for you automatically.

### ==Building and Running a Cargo Project== 

Now let's look at what's different when we build and run the `"Hello, world!"` program with <span style="color:rgb(250, 128, 114)">Cargo!</span> From your `hello_cargo` directory,  build your project by entering the following command:

```bash
$ cargo build
```

This command creates an executable file in *`target/debug/hello_cargo (or target\debug\hello_cargo.exe on Windows)`* rather than in your current directory. Beacuse the default build is a debug build, <span style="color:rgb(250, 128, 114)">Cargo</span> puts the binary in a directory named `debug`. You can run the executable with this com

```
$ ./target/debug/hello_cargo # or .\target\debug\hello_cargo.exe on Windows
Hello, world!
```

If all goes well. `Hello, world!` should print to the terminal. Running `cargo build` for the first time also causes <span style="color:rgb(250, 128, 114)">Cargo</span> to create a new file at the top level: `Cargo.lock`. This file keeps tracks of the exact versions of dependencies in your project. This project doesn't have dependencies, so the file is a bit sparse. You won't ever need to change this file manually <span style="color:rgb(250, 128, 114)">Cargo</span> manages its contents for you.

We just built a project with `cargo build` and ran it with *`target/debug/hello_cargo (or target\debug\hello_cargo.exe on Windows)`* , but we can also use `cargo run` to compile the code and then run the resultant executable all in one command:

```
$ cargo run
    Finished dev [unoptimized + debuginfo] target(s) in 0.0 secs
     Running `target/debug/hello_cargo`
Hello, world!
```

Using `cargo run` is more convenient than having to remember to run `cargo build` and then use the whole path to the binary, so most developers use `cargo run`  (point to be noted that this command will only if at all there is already rust program in the directory, which can be done by using `cargo project name`.)

If you had modified your source code, <span style="color:rgb(250, 128, 114)">Cargo</span> would have rebuilt the project before running it, and you would have seen this output:

```
$ cargo run
   Compiling hello_cargo v0.1.0 (file:///projects/hello_cargo)
    Finished dev [unoptimized + debuginfo] target(s) in 0.33 secs
     Running `target/debug/hello_cargo`
Hello, world!
```

`cargo check` is like Rust, "Did I write this correctly?" without actually creating a runnable program. It compiles your code in memory just to find mistakes (syntax errors, type errors, missing variables, etc.). It doesn't create the final executable file, so it's much faster than cargo `cargo build` or `cargo run`

### ==Building for Release==

When you're working on a Rust project, there are two ways to compile it:

`Development mode (cargo build or cargo run) `:  Creates the executable in `target/debug`.Compiles fast so you can quickly test changes. The program run slower because compiler optimizations (Compiler optimizations are trick the compiler uses to make your program run faster and sometimes use less memory, without changing what the program does) are turned off. Ideal when you're writing code and rebuilding often.

`Release mode (cargo build --release)` : Create the executable in `target/release`. Compiles slower because it enables a lot of optimization (like inlining, loop unrolling, etc.). The program runs much faster because of those optimizations. Ideal when you're:
-  Shipping the final version to users
-  Running benchmarks or performance tests

`(inlining: means the compiler copies the function's code directly into the place it's called, so there's no jump.)
`(Loop unrolling: means the compiler repeats the loop body multiple times inside a single loop iteration to reduce the number of checks and jumps)

To work on any existing projects, you can use the following commands to check out the code using Git, change to the project's directory, and build:

``` shell
$ git clone example.org/someproject
$ cd someproject
$ cargo build
```
