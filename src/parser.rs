use crate::ast::*;

pub fn parse_input(input: &str) -> Result<AST, String> {
    let input = input.trim();

    if input.is_empty() {
        return Err("Empty input".to_string());
    }

    if input.contains('=') && !input.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
        return Err("Invalid assignment".to_string());
    }

    let parts: Vec<&str> = input.split('=').collect();

    if parts.len() == 2 {
        let dest = parts[0].trim().to_string();
        let value_input = parts[1].trim();

        if value_input.ends_with(')') && value_input.contains('(') {
            match parse_call(value_input) {
                Ok(ast) => Ok(AST::Assign {
                    dest,
                    child: Box::new(ast),
                }),
                Err(e) => Err(e),
            }
        } else {
            match parse_argument(value_input) {
                Ok(arg) => Ok(AST::Var {
                    name: dest,
                    value: arg,
                }),
                Err(e) => Err(e),
            }
        }
    } else if input.ends_with(')') && input.contains('(') {
        parse_call(input)
    } else {
        Err("Invalid input format".to_string())
    }
}

fn parse_call(input: &str) -> Result<AST, String> {
    if !input.ends_with(')') {
        return Err("Missing closing parenthesis".to_string());
    }

    let open_paren = input.find('(').ok_or("Missing opening parenthesis")?;
    let name = input[..open_paren].trim().to_string();

    if name.chars().any(|c| !(c.is_alphanumeric() || c == '_')) {
        return Err(format!("Invalid function name: {}", name));
    }
    let args_str = &input[open_paren + 1..input.len() - 1];

    let args = if args_str.trim().is_empty() {
        Vec::new()
    } else {
        let parts: Vec<&str> = args_str.split(',').collect();
        let mut args = Vec::new();
        let mut current_arg = String::new();
        let mut bracket_count = 0;

        for part in parts {
            let part = part.trim();
            if bracket_count == 0 {
                if !current_arg.is_empty() {
                    current_arg.push(',');
                }
                current_arg.push_str(part);
            } else {
                current_arg.push(',');
                current_arg.push_str(part);
            }

            bracket_count += part.matches('[').count() as i32;
            bracket_count -= part.matches(']').count() as i32;

            if bracket_count == 0 && !current_arg.is_empty() {
                args.push(parse_argument(&current_arg)?);
                current_arg.clear();
            }
        }

        if !current_arg.is_empty() {
            args.push(parse_argument(&current_arg)?);
        }

        args
    };

    Ok(AST::Call { name, args })
}

fn parse_argument(input: &str) -> Result<Argument, String> {
    if input.contains('[') && input.ends_with(']') {
        parse_array(input)
    } else if input.starts_with("0x") || input.starts_with("0o") || input.starts_with("0b") {
        parse_number(input)
    } else if input.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        parse_number(input)
    } else if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(Argument::Variable(input.to_string()))
    } else {
        Err(format!("Invalid argument: {}", input))
    }
}

fn parse_array(input: &str) -> Result<Argument, String> {
    let open_bracket = input.find('[').ok_or("Missing opening bracket")?;
    let array_type = &input[..open_bracket];

    let values_str = &input[open_bracket + 1..input.len() - 1];

    if values_str.trim().is_empty() {
        return Ok(Argument::Array(ArgType::U64, [0u8; 64]));
    }

    match array_type {
        "bits" => {
            let values: Vec<u64> = values_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<u64>()
                        .map_err(|_| format!("Invalid bit value: {}", s))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut bytes = [0u8; 64];
            if values.len() <= 64 {
                for (i, &bit) in values.iter().enumerate() {
                    let byte_index = i / 8;
                    let bit_index = i % 8;
                    if bit != 0 {
                        bytes[byte_index] |= 1 << bit_index;
                    }
                }
                Ok(Argument::Array(ArgType::I512, bytes))
            } else {
                let mut current_byte = 0u8;
                let mut byte_index = 0;
                for (i, &bit) in values.iter().enumerate() {
                    if i > 0 && i % 8 == 0 {
                        if byte_index < 64 {
                            bytes[byte_index] = current_byte;
                            byte_index += 1;
                        }
                        current_byte = 0;
                    }
                    if bit != 0 {
                        current_byte |= 1 << (i % 8);
                    }
                }
                if byte_index < 64 {
                    bytes[byte_index] = current_byte;
                }
                Ok(Argument::Array(ArgType::I512, bytes))
            }
        }
        "b" | "w" | "dw" | "qw" => {
            let values = values_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(parse_number)
                .map(|r| {
                    r.and_then(|arg| match arg {
                        Argument::Scalar(val) => Ok(val),
                        _ => Err("Array values must be scalar numbers".to_string()),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let arg_type = ArgType::from_value_count(&values);
            let mut rbytes = [0u8; 64];

            match array_type {
                "b" => {
                    for (i, &val) in values.iter().take(64).enumerate() {
                        rbytes[i] = val as u8;
                    }
                }
                "w" => {
                    for (i, &val) in values.iter().take(32).enumerate() {
                        let bytes = (val as u16).to_le_bytes();
                        rbytes[i * 2..i * 2 + 2].copy_from_slice(&bytes);
                    }
                }
                "dw" => {
                    for (i, &val) in values.iter().take(16).enumerate() {
                        let bytes = (val as u32).to_le_bytes();
                        rbytes[i * 4..i * 4 + 4].copy_from_slice(&bytes);
                    }
                }
                "qw" => {
                    for (i, &val) in values.iter().take(8).enumerate() {
                        let bytes = val.to_le_bytes();
                        rbytes[i * 8..i * 8 + 8].copy_from_slice(&bytes);
                    }
                }
                _ => unreachable!(),
            }
            Ok(Argument::Array(arg_type, rbytes))
        }
        _ => return Err(format!("Unknown array type: {}", array_type)),
    }
}

fn parse_number(input: &str) -> Result<Argument, String> {
    if input.starts_with("0x") {
        let hex_str = &input[2..];
        if hex_str.is_empty() {
            return Err("Empty hex number".to_string());
        }
        u64::from_str_radix(hex_str, 16)
            .map(Argument::Scalar)
            .map_err(|_| format!("Invalid hex number: {}", input))
    } else if input.starts_with("0o") {
        let octal_str = &input[2..];
        if octal_str.is_empty() {
            return Err("Empty octal number".to_string());
        }
        u64::from_str_radix(octal_str, 8)
            .map(Argument::Scalar)
            .map_err(|_| format!("Invalid octal number: {}", input))
    } else if input.starts_with("0b") {
        let binary_str = &input[2..];
        if binary_str.is_empty() {
            return Err("Empty binary number".to_string());
        }
        u64::from_str_radix(binary_str, 2)
            .map(Argument::Scalar)
            .map_err(|_| format!("Invalid binary number: {}", input))
    } else {
        input
            .parse::<u64>()
            .map(Argument::Scalar)
            .map_err(|_| format!("Invalid decimal number: {}", input))
    }
}
