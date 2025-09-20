use std::arch::x86_64::*;
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
pub enum ArgType {
    I256,
    I512,
    U64,
    U32,
    U16,
    U8,
    Ptr,
}

#[derive(Debug)]
pub struct FunctionInfo {
    pub name: String,
    pub arguments: Vec<ArgType>,
    pub return_type: ArgType,
}

#[derive(Debug)]
pub struct FunctionRegistry {
    pub functions: Vec<FunctionInfo>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    pub fn register(&mut self, name: &str, arguments: Vec<ArgType>, return_type: ArgType) {
        self.functions.push(FunctionInfo {
            name: name.to_string(),
            arguments,
            return_type,
        });
    }

    pub fn find(&self, name: &str) -> Option<&FunctionInfo> {
        self.functions.iter().find(|f| f.name == name)
    }
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
    Call { name: String, args: Vec<Argument> },
    Var { name: String, value: Argument },
    Assign { dest: String, child: Box<AST> },
}

impl Argument {
    pub fn to_i256(&self) -> __m256i {
        match self {
            Argument::Array(_, values) => {
                let mut result = [0i32; 8];
                for (i, &val) in values.iter().take(8).enumerate() {
                    result[i] = val as i32;
                }
                unsafe { std::mem::transmute(result) }
            }
            Argument::Scalar(val) => {
                let mut result = [0i32; 8];
                result[0] = *val as i32;
                unsafe { std::mem::transmute(result) }
            }
            _ => unsafe { std::mem::transmute([0i32; 8]) },
        }
    }

    pub fn to_i512(&self) -> __m512i {
        match self {
            Argument::Array(_, values) => {
                let mut result = [0i64; 8];
                for (i, &val) in values.iter().take(8).enumerate() {
                    result[i] = val as i64;
                }
                unsafe { std::mem::transmute(result) }
            }
            Argument::Scalar(val) => {
                let mut result = [0i64; 8];
                result[0] = *val as i64;
                unsafe { std::mem::transmute(result) }
            }
            _ => unsafe { std::mem::transmute([0i64; 8]) },
        }
    }

    pub fn to_u64(&self) -> u64 {
        match self {
            Argument::Scalar(val) => *val,
            Argument::Array(_, values) => values.first().copied().unwrap_or(0),
            _ => 0,
        }
    }

    pub fn to_u32(&self) -> u32 {
        match self {
            Argument::Scalar(val) => *val as u32,
            Argument::Array(_, values) => values.first().map(|&v| v as u32).unwrap_or(0),
            _ => 0,
        }
    }

    pub fn to_u16(&self) -> u16 {
        match self {
            Argument::Scalar(val) => *val as u16,
            Argument::Array(_, values) => values.first().map(|&v| v as u16).unwrap_or(0),
            _ => 0,
        }
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            Argument::Scalar(val) => *val as u8,
            Argument::Array(_, values) => values.first().map(|&v| v as u8).unwrap_or(0),
            _ => 0,
        }
    }
}

pub struct Interpreter {
    pub variables: HashMap<String, Argument>,
    pub function_registry: FunctionRegistry,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut registry = FunctionRegistry::new();

        // Register the AVX functions
        registry.register(
            "_mm256_mask_expand_epi8",
            vec![ArgType::I256, ArgType::I256, ArgType::U8],
            ArgType::I256,
        );
        registry.register(
            "_mm256_mask_expand_epi16",
            vec![ArgType::I256, ArgType::I256, ArgType::U16],
            ArgType::I256,
        );
        registry.register(
            "_mm256_mask_expand_epi32",
            vec![ArgType::I256, ArgType::I256, ArgType::U32],
            ArgType::I256,
        );
        registry.register(
            "_mm256_mask_expand_epi64",
            vec![ArgType::I256, ArgType::I256, ArgType::U64],
            ArgType::I256,
        );

        Self {
            variables: HashMap::new(),
            function_registry: registry,
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
                Argument::Variable(var_name) => self
                    .variables
                    .get(&var_name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", var_name)),
                other => Ok(other),
            })
            .collect::<Result<Vec<_>, _>>()?;

        println!("Resolved arguments: {:?}", resolved_args);
        match name.as_str() {
            "add" => self.execute_add(&resolved_args),
            "test" => Ok(Argument::Scalar(42)),
            "_mm256_mask_expand_epi8" => self.execute_mm256_mask_expand_epi8(&resolved_args),
            "_mm256_mask_expand_epi16" => self.execute_mm256_mask_expand_epi16(&resolved_args),
            "_mm256_mask_expand_epi32" => self.execute_mm256_mask_expand_epi32(&resolved_args),
            "_mm256_mask_expand_epi64" => self.execute_mm256_mask_expand_epi64(&resolved_args),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    fn execute_add(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 2 {
            return Err("add requires exactly 2 arguments".to_string());
        }

        let a = args[0].to_u64();
        let b = args[1].to_u64();

        Ok(Argument::Scalar(a + b))
    }

    fn execute_mm256_mask_expand_epi8(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi8 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u8() as u32;

        let result = unsafe { _mm256_mask_expand_epi8(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi16(&self, args: &[Argument]) -> Result<Argument, String> {
        println!("Executing _mm256_mask_expand_epi16 {:?}", args);
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi16 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u16() as u16;

        let result = unsafe { _mm256_mask_expand_epi16(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi32(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi32 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u32() as u8;

        let result = unsafe { _mm256_mask_expand_epi32(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi64(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi64 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u64() as u8;

        let result = unsafe { _mm256_mask_expand_epi64(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn m256i_to_argument(&self, value: __m256i) -> Argument {
        let array: [i32; 8] = unsafe { std::mem::transmute(value) };
        Argument::Array(AType::DoubleWord, array.iter().map(|&x| x as u64).collect())
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
        input
            .parse::<u64>()
            .map(Argument::Scalar)
            .map_err(|_| format!("Invalid decimal number: {}", input))
    }
}
