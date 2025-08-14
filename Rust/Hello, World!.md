It is traditional when learning a new language to write program that prints the text `hello, world!` to the screen, so we'll do the same here!

### ==Creating a Project Directory==

You'll start by making a directory to store your Rust code. It doesn't matter to Rust where code lives, but for the exercises and projects in this book, we suggest making a <i>projects</i> director your home directory and keeping all your projects there.

Open a terminal and enter the following commands to make a <i>projects</i> directory and a directory for the "Hello, world!" project within the <i>project</i> directory

```powershell
> mkdir "%USERPROFILE%\projects"
> cd /d "%USERPROFILE%\projects"
> mkdir hello_world
> cd hello_world
```

### ==Writing and Running a  Rust Program==

Next, make a new source file and `hello.rs`. Rust files always end with the `.rs` extension. If you're using more than one word in your filename, the convention is to use as underscore to separate them. For example, use `hello_world.rs` rather than `helloworld.rs`.

```rust
fn main() {
	println!("hello, world!");
}
```

Save the file and go back to your terminal window in the directory .
```powershell
> rustc hello.rs
> .\hello
hello, world!
```

### ==Anatomy of  a Rust Program== 

```rust
fn main() {

}
```

These lines define a function named `main`. The `main` function is special: It is always the first code that runs in every executable Rust program. Here, the first line declares a function named `main` that has no parameter and returns nothing. If there were parameters, they would go inside the parentheses `()`.

The function body is wrapped in `{}`. Rust requires curly bracket around all function bodies. It's good style to place the opening curly bracket on the same line as the function declaration, adding one space in between.

The body of the `main` function holds the following code:

```rust
println!("hello, world!");
```

This line does all the work in this little program: it prints text to the screen. there are three important details to notice here.

First, `println!` calls a Rust macro. If it called a function , it would be entered as `println` (without the `!`). Rust macros are a way to write code that generated code to extend Rust syntax. Using a `!` means that  calling a macro instead of a normal function and that macros don't always  always follow the same rules as functions.

Second, you see the `"hello, world!"` string. We pass this string as an argument to  `println!`, and the string is printed to the screen.

Third, we end the line with a semicolon (`;`), which indicates that this expression is over and the next one is ready begin. Most lines of Rust code end with a semicolon.

### ==Compiling and Running Are Separated Steps==

You've just run a newly created program, so let's examine each step in the process.
Before running a Rust program, you must compile it using the Rust complier by entering the `rustc` command and passing it the name of your source file, like this:

```rust 
$ rustc hello.rs
```

If you have a C or C++ background, you'll notice that this is similar to `gcc` or `clang`. After compiling successfully, Rust outputs a binary executable.

Then after we have to run the `.\hello`  to execute the rust file