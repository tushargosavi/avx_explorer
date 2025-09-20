use std::arch::x86_64::*;

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
