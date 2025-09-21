use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

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

pub trait ExecContext {
    fn get_var(&self, name: &str) -> Option<&Argument>;
    fn set_var(&mut self, name: &str, value: Argument);
}

pub trait FunctionInfo {
    fn name(&self) -> &str;
    fn arguments(&self) -> &[ArgType];
    fn return_type(&self) -> ArgType;
    fn execute(&self, ctx: &mut dyn ExecContext, args: &[Argument]) -> Result<Argument, String>;
    fn min_args(&self) -> usize {
        self.arguments().len()
    }
    fn max_args(&self) -> Option<usize> {
        Some(self.arguments().len())
    }
}

#[derive(Debug)]
pub struct Instruction {
    pub name: String,
    pub arguments: Vec<ArgType>,
    pub return_type: ArgType,
    pub exec: fn(&mut dyn ExecContext, &[Argument]) -> Result<Argument, String>,
    pub min_args: usize,
    pub max_args: Option<usize>,
}

impl Instruction {
    pub fn new(
        name: &str,
        arguments: Vec<ArgType>,
        return_type: ArgType,
        exec: fn(&mut dyn ExecContext, &[Argument]) -> Result<Argument, String>,
    ) -> Self {
        let min_args = arguments.len();
        Self {
            name: name.to_string(),
            arguments,
            return_type,
            exec,
            min_args,
            max_args: Some(min_args),
        }
    }

    pub fn with_arg_range(
        name: &str,
        arguments: Vec<ArgType>,
        return_type: ArgType,
        min_args: usize,
        max_args: Option<usize>,
        exec: fn(&mut dyn ExecContext, &[Argument]) -> Result<Argument, String>,
    ) -> Self {
        if let Some(max) = max_args {
            assert!(max >= min_args, "max_args must be >= min_args");
        }

        Self {
            name: name.to_string(),
            arguments,
            return_type,
            exec,
            min_args,
            max_args,
        }
    }
}

impl FunctionInfo for Instruction {
    fn name(&self) -> &str {
        &self.name
    }

    fn arguments(&self) -> &[ArgType] {
        &self.arguments
    }

    fn return_type(&self) -> ArgType {
        self.return_type.clone()
    }

    fn execute(&self, ctx: &mut dyn ExecContext, args: &[Argument]) -> Result<Argument, String> {
        (self.exec)(ctx, args)
    }

    fn min_args(&self) -> usize {
        self.min_args
    }

    fn max_args(&self) -> Option<usize> {
        self.max_args
    }
}

pub struct FunctionRegistry {
    pub functions: HashMap<String, Arc<dyn FunctionInfo>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn register_instruction(&mut self, instruction: Instruction) {
        self.functions
            .insert(instruction.name.clone(), Arc::new(instruction));
    }

    pub fn find(&self, name: &str) -> Option<Arc<dyn FunctionInfo>> {
        self.functions.get(name).cloned()
    }
}

impl ArgType {
    pub fn from_value_count(values: &[u64]) -> Self {
        match values.len() {
            1 => ArgType::U64,
            2..=8 => ArgType::I256, // 256 bits can hold 8 u32 values or 4 u64 values
            _ => ArgType::I512,     // 512 bits for larger arrays
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            ArgType::U8 => 1,
            ArgType::U16 => 2,
            ArgType::U32 => 4,
            ArgType::U64 => 8,
            ArgType::I256 => 32,
            ArgType::I512 => 64,
            ArgType::Ptr => 8,
        }
    }

    pub fn vector_byte_len(&self) -> usize {
        match self {
            ArgType::I256 => 32,
            ArgType::I512 => 64,
            // Keep typed arrays at the full backing buffer size used throughout the
            // interpreter for consistent display helpers.
            ArgType::U8 | ArgType::U16 | ArgType::U32 | ArgType::U64 => 64,
            ArgType::Ptr => 8,
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Argument {
    Array(ArgType, [u8; 64]),
    Memory(Vec<u8>),
    Scalar(u64),
    ScalarTyped(ArgType, u64),
    Variable(String),
}

impl fmt::Debug for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Argument::Array(arg_type, bytes) => {
                let len = arg_type.vector_byte_len().min(bytes.len());
                let slice = &bytes[..len];
                f.debug_tuple("Array")
                    .field(arg_type)
                    .field(&slice)
                    .finish()
            }
            Argument::Memory(bytes) => f.debug_tuple("Memory").field(bytes).finish(),
            Argument::Scalar(val) => f.debug_tuple("Scalar").field(val).finish(),
            Argument::ScalarTyped(arg_type, val) => f
                .debug_tuple("ScalarTyped")
                .field(arg_type)
                .field(val)
                .finish(),
            Argument::Variable(name) => f.debug_tuple("Variable").field(name).finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AST {
    Call { name: String, args: Vec<Argument> },
    Var { name: String, value: Argument },
    Assign { dest: String, child: Box<AST> },
}

impl AST {
    pub fn validate(&self, registry: &FunctionRegistry) -> Result<(), String> {
        match self {
            AST::Call { name, args } => {
                let func = registry
                    .find(name)
                    .ok_or_else(|| format!("Unknown function: {}", name))?;
                let arg_count = args.len();
                let min_args = func.min_args();
                if arg_count < min_args {
                    return Err(format!(
                        "Function '{}' expects at least {} argument(s), but {} provided",
                        name, min_args, arg_count
                    ));
                }
                if let Some(max_args) = func.max_args() {
                    if arg_count > max_args {
                        return Err(format!(
                            "Function '{}' expects at most {} argument(s), but {} provided",
                            name, max_args, arg_count
                        ));
                    }
                }
                Ok(())
            }
            AST::Var { .. } => Ok(()),
            AST::Assign { child, .. } => child.validate(registry),
        }
    }
}

impl Argument {
    pub fn to_i256(&self) -> __m256i {
        match self {
            Argument::Array(arg_type, bytes) => match arg_type {
                ArgType::I256 => {
                    let mut result = [0i32; 8];
                    for i in 0..8 {
                        let start = i * 4;
                        if start + 4 <= bytes.len() {
                            result[i] = i32::from_le_bytes([
                                bytes[start],
                                bytes[start + 1],
                                bytes[start + 2],
                                bytes[start + 3],
                            ]);
                        }
                    }
                    unsafe { std::mem::transmute(result) }
                }
                _ => {
                    let mut result = [0i32; 8];
                    result[0] = self.to_u32() as i32;
                    unsafe { std::mem::transmute(result) }
                }
            },
            Argument::Scalar(val) => {
                let mut result = [0i32; 8];
                result[0] = *val as i32;
                unsafe { std::mem::transmute(result) }
            }
            Argument::ScalarTyped(_, val) => {
                let mut result = [0i32; 8];
                result[0] = *val as i32;
                unsafe { std::mem::transmute(result) }
            }
            _ => unsafe { std::mem::transmute([0i32; 8]) },
        }
    }

    pub fn to_i512(&self) -> __m512i {
        match self {
            Argument::Array(arg_type, bytes) => match arg_type {
                ArgType::I512 => {
                    let mut result = [0i64; 8];
                    for i in 0..8 {
                        let start = i * 8;
                        if start + 8 <= bytes.len() {
                            result[i] = i64::from_le_bytes([
                                bytes[start],
                                bytes[start + 1],
                                bytes[start + 2],
                                bytes[start + 3],
                                bytes[start + 4],
                                bytes[start + 5],
                                bytes[start + 6],
                                bytes[start + 7],
                            ]);
                        }
                    }
                    unsafe { std::mem::transmute(result) }
                }
                _ => {
                    let mut result = [0i64; 8];
                    result[0] = self.to_u64() as i64;
                    unsafe { std::mem::transmute(result) }
                }
            },
            Argument::Scalar(val) => {
                let mut result = [0i64; 8];
                result[0] = *val as i64;
                unsafe { std::mem::transmute(result) }
            }
            Argument::ScalarTyped(_, val) => {
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
            Argument::ScalarTyped(_, val) => *val,
            Argument::Array(_, bytes) => {
                if bytes.len() >= 8 {
                    u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ])
                } else {
                    let mut temp = [0u8; 8];
                    temp[..bytes.len()].copy_from_slice(bytes);
                    u64::from_le_bytes(temp)
                }
            }
            _ => 0,
        }
    }

    pub fn to_u32(&self) -> u32 {
        match self {
            Argument::Scalar(val) => *val as u32,
            Argument::ScalarTyped(_, val) => *val as u32,
            Argument::Array(_, bytes) => {
                if bytes.len() >= 4 {
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    let mut temp = [0u8; 4];
                    temp[..bytes.len()].copy_from_slice(bytes);
                    u32::from_le_bytes(temp)
                }
            }
            _ => 0,
        }
    }

    pub fn to_u16(&self) -> u16 {
        match self {
            Argument::Scalar(val) => *val as u16,
            Argument::ScalarTyped(_, val) => *val as u16,
            Argument::Array(_, bytes) => {
                if bytes.len() >= 2 {
                    u16::from_le_bytes([bytes[0], bytes[1]])
                } else if bytes.len() == 1 {
                    u16::from(bytes[0])
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            Argument::Scalar(val) => *val as u8,
            Argument::ScalarTyped(_, val) => *val as u8,
            Argument::Array(_, bytes) => bytes.first().copied().unwrap_or(0),
            _ => 0,
        }
    }
}
