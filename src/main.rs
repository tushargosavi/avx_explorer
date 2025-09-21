mod ast;
mod avx2;
mod avx_512;
mod bmi2;
mod display;
mod interpreter;
mod parser;
use parser::parse_input;

use crate::interpreter::Interpreter;

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

#[cfg(test)]
mod tests;

fn main() {
    let mut interpreter = Interpreter::new();
    let mut args = std::env::args().skip(1);

    let file_arg = match args.next() {
        Some(arg) if arg == "-h" || arg == "--help" => {
            print_usage();
            return;
        }
        Some(arg) if arg == "-f" || arg == "--file" => match args.next() {
            Some(path) => Some(path),
            None => {
                eprintln!("Expected a file path after {}.", arg);
                std::process::exit(1);
            }
        },
        Some(arg) if arg.starts_with('-') => {
            eprintln!("Unknown option: {}", arg);
            print_usage();
            std::process::exit(1);
        }
        Some(arg) => Some(arg),
        None => None,
    };

    if args.next().is_some() {
        eprintln!("Too many arguments provided.");
        print_usage();
        std::process::exit(1);
    }

    if let Some(path) = file_arg {
        if let Err(err) = run_file(&mut interpreter, &path) {
            eprintln!("Failed to process '{}': {}", path, err);
            std::process::exit(1);
        }
    } else {
        run_repl(&mut interpreter);
    }
}

fn print_usage() {
    println!("Usage: avx_explorer [--file <path>]");
    println!();
    println!("When no file is provided the interactive REPL is started.");
}

fn run_repl(interpreter: &mut Interpreter) {
    println!("AVX2/AVX512 Simulator REPL");
    println!("Type 'exit' to quit");

    loop {
        print!("prompt> ");
        if let Err(err) = io::stdout().flush() {
            eprintln!("Failed to flush stdout: {}", err);
        }

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {
                if !handle_line(interpreter, &input) {
                    break;
                }
            }
            Err(err) => {
                eprintln!("Error reading input: {}", err);
                break;
            }
        }
    }
}

fn run_file(interpreter: &mut Interpreter, path: &str) -> io::Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if !handle_line(interpreter, &line) {
            break;
        }
    }

    Ok(())
}

fn handle_line(interpreter: &mut Interpreter, input: &str) -> bool {
    let input = input.trim();

    if input.is_empty() {
        return true;
    }

    if input == "exit" {
        return false;
    }

    match parse_input(input) {
        Ok(ast) => match interpreter.execute(ast) {
            Ok(result) => println!("{:?}", result),
            Err(e) => println!("Error: {}", e),
        },
        Err(e) => println!("Parse error: {}", e),
    }

    true
}
