mod simple_parser;

use simple_parser::{parse_input, Interpreter};

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
        std::io::stdin()
            .read_line(&mut input)
            .unwrap();

        let input = input.trim();
        if input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        match parse_input(input) {
            Ok(ast) => match interpreter.execute(ast) {
                Ok(result) => println!("{:?}", result),
                Err(e) => println!("Error: {}", e),
            },
            Err(e) => println!("Parse error: {}", e),
        }
    }
}
