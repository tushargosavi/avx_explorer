use crate::ast::*;
use std::arch::x86_64::*;
use std::collections::HashMap;

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

        // Register print function (variable arguments)
        registry.register("print", vec![], ArgType::U64);

        Self {
            variables: HashMap::new(),
            function_registry: registry,
        }
    }

    pub fn execute(&mut self, ast: AST) -> Result<Argument, String> {
        let result = match ast {
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
        };

        // Store result in _res for all statements
        if let Ok(ref result) = result {
            self.variables.insert("_res".to_string(), result.clone());
        }

        result
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

        match name.as_str() {
            "add" => self.execute_add(&resolved_args),
            "test" => Ok(Argument::Scalar(42)),
            "print" => self.execute_print(&resolved_args),
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

    fn execute_print(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.is_empty() {
            // Print _res variable by default
            if let Some(res_arg) = self.variables.get("_res") {
                self.display_argument(res_arg);
            } else {
                println!("_res is undefined");
            }
        } else {
            // Print each argument
            for arg in args {
                self.display_argument(arg);
            }
        }
        Ok(Argument::Scalar(0)) // Return 0 as success
    }

    fn display_argument(&self, arg: &Argument) {
        match arg {
            Argument::Scalar(val) => println!("{}", val),
            Argument::Array(atype, bytes) => match atype {
                AType::Bit => {
                    let bit_count = bytes.iter().map(|&b| b.count_ones()).sum::<u32>();
                    println!("bits[{} bits]", bit_count);
                }
                AType::Byte => {
                    let values_str: Vec<String> =
                        bytes.iter().take(32).map(|v| v.to_string()).collect();
                    println!("b[{}]", values_str.join(", "));
                }
                AType::Word => {
                    let words: Vec<u16> = bytes
                        .chunks(2)
                        .map(|chunk| u16::from_le_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]))
                        .take(32)
                        .collect();
                    let values_str: Vec<String> = words.iter().map(|v| v.to_string()).collect();
                    println!("w[{}]", values_str.join(", "));
                }
                AType::DoubleWord => {
                    let dwords: Vec<u32> = bytes
                        .chunks(4)
                        .map(|chunk| {
                            u32::from_le_bytes([
                                chunk[0],
                                *chunk.get(1).unwrap_or(&0),
                                *chunk.get(2).unwrap_or(&0),
                                *chunk.get(3).unwrap_or(&0),
                            ])
                        })
                        .take(16)
                        .collect();
                    let values_str: Vec<String> = dwords.iter().map(|v| v.to_string()).collect();
                    println!("dw[{}]", values_str.join(", "));
                }
                AType::QuadWord => {
                    let qwords: Vec<u64> = bytes
                        .chunks(8)
                        .map(|chunk| {
                            u64::from_le_bytes([
                                chunk[0],
                                *chunk.get(1).unwrap_or(&0),
                                *chunk.get(2).unwrap_or(&0),
                                *chunk.get(3).unwrap_or(&0),
                                *chunk.get(4).unwrap_or(&0),
                                *chunk.get(5).unwrap_or(&0),
                                *chunk.get(6).unwrap_or(&0),
                                *chunk.get(7).unwrap_or(&0),
                            ])
                        })
                        .take(8)
                        .collect();
                    let values_str: Vec<String> = qwords.iter().map(|v| v.to_string()).collect();
                    println!("qw[{}]", values_str.join(", "));
                }
            },
            Argument::Variable(name) => {
                if let Some(var_arg) = self.variables.get(name) {
                    self.display_argument(var_arg);
                } else {
                    println!("undefined variable: {}", name);
                }
            }
            Argument::Memory(bytes) => println!("memory[{} bytes]", bytes.len()),
        }
    }

    fn execute_mm256_mask_expand_epi8(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi8 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let k = args[1].to_u8() as u32;
        let a = args[2].to_i256();

        let result = unsafe { _mm256_mask_expand_epi8(src, k, a) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi16(&self, args: &[Argument]) -> Result<Argument, String> {
        println!("Executing _mm256_mask_expand_epi16 {:?}", args);
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi16 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let k = args[1].to_u16();
        let a = args[2].to_i256();

        let result = unsafe { _mm256_mask_expand_epi16(src, k, a) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi32(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi32 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let k = args[1].to_u32() as u8;
        let a = args[2].to_i256();

        let result = unsafe { _mm256_mask_expand_epi32(src, k, a) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi64(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi64 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let k = args[1].to_u64() as u8;
        let a = args[2].to_i256();

        println!("src {:?} k {:?} a {:?}", src, k, a);
        let result = unsafe { _mm256_mask_expand_epi64(src, k, a) };
        println!("src {:?} k {:?} a {:?} res {:?}", src, k, a, result);
        Ok(self.m256i_to_argument(result))
    }

    fn m256i_to_argument(&self, value: __m256i) -> Argument {
        let array: [i32; 8] = unsafe { std::mem::transmute(value) };
        let mut bytes = [0u8; 64];
        for (i, &val) in array.iter().enumerate() {
            let val_bytes = val.to_le_bytes();
            let start = i * 4;
            bytes[start..start + 4].copy_from_slice(&val_bytes);
        }
        Argument::Array(AType::DoubleWord, bytes)
    }
}
