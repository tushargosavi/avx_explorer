use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum AType {
    Bit,
    Byte,
    Word,
    DoubleWord,
    QuadWord,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Argument {
    Array(AType, Vec<u64>),
    Memory(Vec<u8>),
    Scalar(u64),
    Variable(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AST {
    Call {
        name: String,
        args: Vec<Argument>,
    },
    Var {
        name: String,
        value: Argument,
    },
    Assign {
        dest: String,
        child: Box<AST>,
    },
}

impl Argument {
    pub fn to_i256(&self) -> Option<[i32; 8]> {
        match self {
            Argument::Array(_, values) => {
                let mut result = [0i32; 8];
                for (i, &val) in values.iter().take(8).enumerate() {
                    result[i] = val as i32;
                }
                Some(result)
            }
            Argument::Scalar(val) => {
                let mut result = [0i32; 8];
                result[0] = *val as i32;
                Some(result)
            }
            _ => None,
        }
    }

    pub fn to_i512(&self) -> Option<[i64; 8]> {
        match self {
            Argument::Array(_, values) => {
                let mut result = [0i64; 8];
                for (i, &val) in values.iter().take(8).enumerate() {
                    result[i] = val as i64;
                }
                Some(result)
            }
            Argument::Scalar(val) => {
                let mut result = [0i64; 8];
                result[0] = *val as i64;
                Some(result)
            }
            _ => None,
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Argument::Scalar(val) => Some(*val),
            Argument::Array(_, values) => values.first().copied(),
            _ => None,
        }
    }

    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Argument::Scalar(val) => Some(*val as u32),
            Argument::Array(_, values) => values.first().map(|&v| v as u32),
            _ => None,
        }
    }

    pub fn to_u16(&self) -> Option<u16> {
        match self {
            Argument::Scalar(val) => Some(*val as u16),
            Argument::Array(_, values) => values.first().map(|&v| v as u16),
            _ => None,
        }
    }
}

pub struct Interpreter {
    pub variables: HashMap<String, Argument>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn execute(&mut self, ast: AST) -> Result<Argument, String> {
        match ast {
            AST::Call { name, args } => self.execute_call(name, args),
            AST::Var { name, value } => {
                self.variables.insert(name.clone(), value.clone());
                Ok(value)
            }
            AST::Assign { dest, child } => {
                let result = self.execute(*child)?;
                self.variables.insert(dest, result.clone());
                Ok(result)
            }
        }
    }

    fn execute_call(&mut self, name: String, args: Vec<Argument>) -> Result<Argument, String> {
        let resolved_args: Vec<Argument> = args
            .into_iter()
            .map(|arg| match arg {
                Argument::Variable(var_name) => {
                    self.variables.get(&var_name).cloned()
                        .ok_or_else(|| format!("Undefined variable: {}", var_name))
                }
                other => Ok(other),
            })
            .collect::<Result<Vec<_>, _>>()?;

        match name.as_str() {
            "add" => self.execute_add(&resolved_args),
            "test" => Ok(Argument::Scalar(42)),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    fn execute_add(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 2 {
            return Err("add requires exactly 2 arguments".to_string());
        }

        let a = args[0].to_u64().ok_or("First argument must be convertible to u64")?;
        let b = args[1].to_u64().ok_or("Second argument must be convertible to u64")?;

        Ok(Argument::Scalar(a + b))
    }
}

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
                    child: Box::new(ast)
                }),
                Err(e) => Err(e),
            }
        } else {
            match parse_argument(value_input) {
                Ok(arg) => Ok(AST::Var {
                    name: dest,
                    value: arg
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

    let atype = match array_type {
        "bits" => AType::Bit,
        "b" => AType::Byte,
        "w" => AType::Word,
        "dw" => AType::DoubleWord,
        "qw" => AType::QuadWord,
        _ => return Err(format!("Unknown array type: {}", array_type)),
    };

    let values_str = &input[open_bracket + 1..input.len() - 1];

    if values_str.trim().is_empty() {
        return Ok(Argument::Array(atype, Vec::new()));
    }

    let values = values_str.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(parse_number)
        .map(|r| r.and_then(|arg| match arg {
            Argument::Scalar(val) => Ok(val),
            _ => Err("Array values must be scalar numbers".to_string()),
        }))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Argument::Array(atype, values))
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
        input.parse::<u64>()
            .map(Argument::Scalar)
            .map_err(|_| format!("Invalid decimal number: {}", input))
    }
}