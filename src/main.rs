mod ast;
mod avx2;
mod bmi2;
mod display;
mod interpreter;
mod parser;
mod sse2;
use parser::parse_input;

use crate::interpreter::Interpreter;

#[cfg(test)]
mod tests;

fn main() {
    let mut interpreter = Interpreter::new();

    println!("AVX2/AVX512 Simulator REPL");
    println!("Type 'exit' to quit");

    loop {
        print!("prompt> ");
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        let read = std::io::stdin().read_line(&mut input).unwrap();

        if read == 0 {
            break;
        }

        let input = input.trim();
        if input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        match parse_input(input) {
            Ok(ast) => {
                //println!("ast {:?}", ast);
                match interpreter.execute(ast) {
                    Ok(result) => println!("{:?}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }
}
