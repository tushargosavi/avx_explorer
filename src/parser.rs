use crate::ast::*;

const ARRAY_TYPES: &[&str] = &["bits", "b", "w", "dw", "qw"];

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
            if let Some((name, access)) = parse_memory_target(&dest)? {
                let value = parse_argument(value_input)?;
                return Ok(AST::MemoryStore {
                    name,
                    access,
                    value,
                });
            }
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
    } else if input.chars().all(|c| c.is_alphanumeric() || c == '_') && !input.is_empty() {
        // Variable lookup - just the variable name
        Ok(AST::VarLookup {
            name: input.to_string(),
        })
    } else if let Some((name, access)) = parse_memory_target(input)? {
        Ok(AST::MemorySliceLookup { name, access })
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
    // memory constructor: mem[SIZE] or mem[values...]
    if input.starts_with('"') && input.ends_with('"') {
        return parse_string_literal(input);
    }
    if input.starts_with("mem[") && input.ends_with(']') {
        return parse_memory(input);
    }
    if input.starts_with("zero[") && input.ends_with(']') {
        return parse_zero(input);
    }
    if let Some(arg) = try_parse_memory_slice_argument(input)? {
        return Ok(arg);
    }
    if input.starts_with('[') && input.ends_with(']') {
        return parse_memory_literal(input);
    }
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

fn parse_string_literal(input: &str) -> Result<Argument, String> {
    if input.len() < 2 || !input.starts_with('"') || !input.ends_with('"') {
        return Err("Invalid string literal".to_string());
    }

    let mut bytes = Vec::new();
    let mut chars = input[1..input.len() - 1].chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let escaped = chars
                .next()
                .ok_or_else(|| "Incomplete escape sequence in string literal".to_string())?;
            match escaped {
                '\\' => bytes.push(b'\\'),
                '"' => bytes.push(b'"'),
                'n' => bytes.push(b'\n'),
                'r' => bytes.push(b'\r'),
                't' => bytes.push(b'\t'),
                '0' => bytes.push(0),
                'x' => {
                    let hi = chars
                        .next()
                        .ok_or_else(|| "Incomplete hex escape in string literal".to_string())?;
                    let lo = chars
                        .next()
                        .ok_or_else(|| "Incomplete hex escape in string literal".to_string())?;
                    let mut hex = String::new();
                    hex.push(hi);
                    hex.push(lo);
                    let value = u8::from_str_radix(&hex, 16)
                        .map_err(|_| format!("Invalid hex escape: \\x{}", hex))?;
                    bytes.push(value);
                }
                other => {
                    return Err(format!("Unsupported escape sequence: \\{}", other));
                }
            }
        } else {
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
    }

    Ok(Argument::Memory(bytes))
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

            let arg_type = match array_type {
                "b" => ArgType::U8,
                "w" | "dw" | "qw" => ArgType::I256,
                _ => ArgType::from_value_count(&values),
            };
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

fn parse_memory(input: &str) -> Result<Argument, String> {
    // Format: mem[SIZE] or mem[val, val, ...]
    if !input.starts_with("mem[") || !input.ends_with(']') {
        return Err("Invalid memory initializer".to_string());
    }
    let size_str = &input[4..input.len() - 1];
    if size_str.trim().is_empty() {
        return Err("mem[] requires a size".to_string());
    }
    if size_str.contains(',') {
        let bytes = parse_memory_values(size_str)?;
        Ok(Argument::Memory(bytes))
    } else {
        // Reuse number parser so hex/bin/dec work; expect Scalar or ScalarTyped
        let size_arg = parse_number(size_str)?;
        let size = match size_arg {
            Argument::Scalar(v) => v as usize,
            Argument::ScalarTyped(_, v) => v as usize,
            _ => return Err("mem[size] size must be a scalar number".to_string()),
        };
        Ok(Argument::Memory(vec![0u8; size]))
    }
}

fn parse_zero(input: &str) -> Result<Argument, String> {
    if !input.starts_with("zero[") || !input.ends_with(']') {
        return Err("Invalid zero initializer".to_string());
    }
    let size_str = &input[5..input.len() - 1];
    if size_str.trim().is_empty() {
        return Err("zero[] requires a size".to_string());
    }
    let size_arg = parse_number(size_str.trim())?;
    let size = match size_arg {
        Argument::Scalar(v) => v as usize,
        Argument::ScalarTyped(_, v) => v as usize,
        _ => return Err("zero[size] size must be a scalar number".to_string()),
    };
    Ok(Argument::Memory(vec![0u8; size]))
}

fn parse_memory_literal(input: &str) -> Result<Argument, String> {
    if !input.starts_with('[') || !input.ends_with(']') {
        return Err("Invalid memory literal".to_string());
    }
    let values_str = &input[1..input.len() - 1];
    if values_str.trim().is_empty() {
        return Ok(Argument::Memory(Vec::new()));
    }
    let bytes = parse_memory_values(values_str)?;
    Ok(Argument::Memory(bytes))
}

fn parse_memory_values(values_str: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut current_type = ArgType::U8;
    for raw in values_str.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let number = parse_number(token)?;
        match number {
            Argument::ScalarTyped(arg_type, value) => {
                current_type = arg_type;
                append_value_as_type(&mut bytes, &current_type, value)?;
            }
            Argument::Scalar(value) => {
                append_value_as_type(&mut bytes, &current_type, value)?;
            }
            _ => {
                return Err("Memory initializers must be scalar numbers".to_string());
            }
        }
    }
    Ok(bytes)
}

fn append_value_as_type(bytes: &mut Vec<u8>, arg_type: &ArgType, value: u64) -> Result<(), String> {
    let width = match arg_type {
        ArgType::U8 => 1,
        ArgType::U16 => 2,
        ArgType::U32 => 4,
        ArgType::U64 => 8,
        _ => {
            return Err(
                "Only unsigned scalar types (u8/u16/u32/u64) are supported in memory initializers"
                    .to_string(),
            );
        }
    };
    let le_bytes = value.to_le_bytes();
    bytes.extend_from_slice(&le_bytes[..width]);
    Ok(())
}

fn try_parse_memory_slice_argument(input: &str) -> Result<Option<Argument>, String> {
    if !input.ends_with(']') {
        return Ok(None);
    }
    let open = match input.find('[') {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let prefix = input[..open].trim();
    if prefix.is_empty() {
        return Ok(None);
    }
    if ARRAY_TYPES.contains(&prefix) {
        return Ok(None);
    }
    if !prefix
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Ok(None);
    }
    let inner = &input[open + 1..input.len() - 1];
    if inner.contains(',') {
        return Ok(None);
    }
    let access = parse_memory_access(inner)?;
    Ok(Some(Argument::MemorySlice {
        name: prefix.to_string(),
        access,
    }))
}

fn parse_memory_target(input: &str) -> Result<Option<(String, MemoryAccess)>, String> {
    if !input.ends_with(']') {
        return Ok(None);
    }
    let open = match input.find('[') {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let name = input[..open].trim();
    if name.is_empty() {
        return Ok(None);
    }
    if ARRAY_TYPES.contains(&name) {
        return Ok(None);
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Ok(None);
    }
    let inner = &input[open + 1..input.len() - 1];
    if inner.trim().is_empty() {
        return Err("Memory slice requires an index or range".to_string());
    }
    if inner.contains(',') {
        return Ok(None);
    }
    let access = parse_memory_access(inner)?;
    Ok(Some((name.to_string(), access)))
}

fn parse_memory_access(input: &str) -> Result<MemoryAccess, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Memory slice requires an index or range".to_string());
    }
    if let Some(range_pos) = trimmed.find("..") {
        let start_str = trimmed[..range_pos].trim();
        let end_str = trimmed[range_pos + 2..].trim();
        if end_str.is_empty() {
            return Err("Memory slice requires an end for ranges".to_string());
        }
        let start = if start_str.is_empty() {
            0
        } else {
            parse_slice_bound(start_str)?
        };
        let end = parse_slice_bound(end_str)?;
        Ok(MemoryAccess::Range { start, end })
    } else {
        let idx = parse_slice_bound(trimmed)?;
        Ok(MemoryAccess::Index(idx))
    }
}

fn parse_slice_bound(token: &str) -> Result<usize, String> {
    let arg = parse_number(token)?;
    match arg {
        Argument::Scalar(v) => Ok(v as usize),
        Argument::ScalarTyped(_, v) => Ok(v as usize),
        _ => Err("Slice bounds must be scalar numbers".to_string()),
    }
}

fn parse_number(input: &str) -> Result<Argument, String> {
    // Allow optional unsigned integer suffix: u8/u16/u32/u64
    let (num_part, suffix) = if let Some(pos) = input.rfind(|c: char| c == 'u') {
        (&input[..pos], Some(&input[pos..]))
    } else {
        (input, None)
    };
    let typed = match suffix {
        Some("u8") => Some(ArgType::U8),
        Some("u16") => Some(ArgType::U16),
        Some("u32") => Some(ArgType::U32),
        Some("u64") => Some(ArgType::U64),
        Some(_) => return Err(format!("Invalid numeric suffix: {}", suffix.unwrap())),
        None => None,
    };

    if num_part.starts_with("0x") {
        let cleaned = num_part[2..].replace('_', "");
        if cleaned.is_empty() {
            return Err("Empty hex number".to_string());
        }
        let val = u64::from_str_radix(&cleaned, 16)
            .map_err(|_| format!("Invalid hex number: {}", input))?;
        Ok(match typed {
            Some(t) => Argument::ScalarTyped(t, val),
            None => Argument::Scalar(val),
        })
    } else if num_part.starts_with("0o") {
        let cleaned = num_part[2..].replace('_', "");
        if cleaned.is_empty() {
            return Err("Empty octal number".to_string());
        }
        let val = u64::from_str_radix(&cleaned, 8)
            .map_err(|_| format!("Invalid octal number: {}", input))?;
        Ok(match typed {
            Some(t) => Argument::ScalarTyped(t, val),
            None => Argument::Scalar(val),
        })
    } else if num_part.starts_with("0b") {
        let cleaned = num_part[2..].replace('_', "");
        if cleaned.is_empty() {
            return Err("Empty binary number".to_string());
        }
        let val = u64::from_str_radix(&cleaned, 2)
            .map_err(|_| format!("Invalid binary number: {}", input))?;
        Ok(match typed {
            Some(t) => Argument::ScalarTyped(t, val),
            None => Argument::Scalar(val),
        })
    } else {
        let cleaned = num_part.replace('_', "");
        if cleaned.is_empty() {
            return Err("Empty decimal number".to_string());
        }
        let val = cleaned
            .parse::<u64>()
            .map_err(|_| format!("Invalid decimal number: {}", input))?;
        Ok(match typed {
            Some(t) => Argument::ScalarTyped(t, val),
            None => Argument::Scalar(val),
        })
    }
}
