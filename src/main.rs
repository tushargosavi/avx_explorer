mod ast;
mod avx2;
mod avx_512;
mod bmi2;
mod display;
mod interpreter;
mod parser;
mod sse2;
use parser::parse_input;

use crate::interpreter::Interpreter;

use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

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

    let mut editor = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(err) => {
            eprintln!("Failed to initialize line editor: {}", err);
            std::process::exit(1);
        }
    };

    let _ = editor.load_history(".avx_explorer_history");

    loop {
        match editor.readline("prompt> ") {
            Ok(line) => {
                let trimmed = line.trim();

                if trimmed.is_empty() {
                    continue;
                }

                if editor.add_history_entry(trimmed).is_err() {
                    eprintln!("Warning: failed to add line to history");
                }

                if !handle_line(interpreter, trimmed) {
                    break;
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("Error reading input: {}", err);
                break;
            }
        }
    }

    if let Err(err) = editor.save_history(".avx_explorer_history") {
        eprintln!("Warning: failed to save history: {}", err);
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
