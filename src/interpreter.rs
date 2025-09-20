use crate::ast::*;
use crate::avx2::register_avx2_instructions;
use crate::bmi2::register_bmi2_instructions;
use std::arch::x86_64::*;
use std::collections::HashMap;

pub struct Interpreter {
    pub variables: HashMap<String, Argument>,
    pub function_registry: FunctionRegistry,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut registry = FunctionRegistry::new();

        // Register core and AVX functions
        registry.register_instruction(Instruction::new(
            "add",
            vec![ArgType::U64, ArgType::U64],
            ArgType::U64,
            |_, args| {
                if args.len() != 2 {
                    return Err("add requires exactly 2 arguments".to_string());
                }
                Ok(Argument::Scalar(args[0].to_u64() + args[1].to_u64()))
            },
        ));

        registry.register_instruction(Instruction::new("test", vec![], ArgType::U64, |_, args| {
            if let Some(first) = args.get(0) {
                Ok(first.clone())
            } else {
                Ok(Argument::Scalar(42))
            }
        }));

        registry.register_instruction(Instruction::new(
            "print",
            vec![],
            ArgType::U64,
            |ctx, args| {
                if args.is_empty() {
                    if let Some(res) = ctx.get_var("_res") {
                        display_argument_simple(res);
                    } else {
                        println!("_res is undefined");
                    }
                } else {
                    for arg in args {
                        display_argument_simple(arg);
                    }
                }
                Ok(Argument::Scalar(0))
            },
        ));

        // print_hex(value [, chunk_bits]) where chunk_bits in {8,16,32,64}; default 32
        registry.register_instruction(Instruction::new(
            "print_hex",
            vec![],
            ArgType::U64,
            |_, args| {
                let (value_arg, chunk_bits_opt) = match args.len() {
                    0 => return Err("print_hex requires at least 1 argument".to_string()),
                    1 => (&args[0], None),
                    _ => (&args[0], Some(args[1].to_u64())),
                };
                print_hex(value_arg, chunk_bits_opt)?;
                Ok(Argument::Scalar(0))
            },
        ));

        // print_dec(value)
        registry.register_instruction(Instruction::new(
            "print_dec",
            vec![],
            ArgType::U64,
            |_, args| {
                if args.is_empty() {
                    return Err("print_dec requires 1 argument".to_string());
                }
                print_dec(&args[0])?;
                Ok(Argument::Scalar(0))
            },
        ));

        // print_bin(value)
        registry.register_instruction(Instruction::new(
            "print_bin",
            vec![],
            ArgType::U64,
            |_, args| {
                if args.is_empty() {
                    return Err("print_bin requires 1 argument".to_string());
                }
                print_bin(&args[0])?;
                Ok(Argument::Scalar(0))
            },
        ));

        registry.register_instruction(Instruction::new(
            "_mm256_mask_expand_epi8",
            vec![ArgType::I256, ArgType::I256, ArgType::U8],
            ArgType::I256,
            |_, args| {
                if args.len() != 3 {
                    return Err("_mm256_mask_expand_epi8 requires exactly 3 arguments".to_string());
                }
                let src = args[0].to_i256();
                let k = args[1].to_u8() as u32;
                let a = args[2].to_i256();
                let result = unsafe { _mm256_mask_expand_epi8(src, k, a) };
                Ok(m256i_to_argument(result))
            },
        ));

        registry.register_instruction(Instruction::new(
            "_mm256_mask_expand_epi16",
            vec![ArgType::I256, ArgType::I256, ArgType::U16],
            ArgType::I256,
            |_, args| {
                if args.len() != 3 {
                    return Err("_mm256_mask_expand_epi16 requires exactly 3 arguments".to_string());
                }
                let src = args[0].to_i256();
                let k = args[1].to_u16();
                let a = args[2].to_i256();
                let result = unsafe { _mm256_mask_expand_epi16(src, k, a) };
                Ok(m256i_to_argument(result))
            },
        ));

        registry.register_instruction(Instruction::new(
            "_mm256_mask_expand_epi32",
            vec![ArgType::I256, ArgType::I256, ArgType::U32],
            ArgType::I256,
            |_, args| {
                if args.len() != 3 {
                    return Err("_mm256_mask_expand_epi32 requires exactly 3 arguments".to_string());
                }
                let src = args[0].to_i256();
                let k = args[1].to_u32() as u8;
                let a = args[2].to_i256();
                let result = unsafe { _mm256_mask_expand_epi32(src, k, a) };
                Ok(m256i_to_argument(result))
            },
        ));

        registry.register_instruction(Instruction::new(
            "_mm256_mask_expand_epi64",
            vec![ArgType::I256, ArgType::I256, ArgType::U64],
            ArgType::I256,
            |_, args| {
                if args.len() != 3 {
                    return Err("_mm256_mask_expand_epi64 requires exactly 3 arguments".to_string());
                }
                let src = args[0].to_i256();
                let k = args[1].to_u64() as u8;
                let a = args[2].to_i256();
                let result = unsafe { _mm256_mask_expand_epi64(src, k, a) };
                Ok(m256i_to_argument(result))
            },
        ));

        // Register AVX2 intrinsic-backed instructions
        register_avx2_instructions(&mut registry);
        // Register BMI2 intrinsic-backed instructions
        register_bmi2_instructions(&mut registry);

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

        if let Some(func) = self.function_registry.find(&name) {
            let ctx: &mut dyn ExecContext = self;
            func.execute(ctx, &resolved_args)
        } else {
            Err(format!("Unknown function: {}", name))
        }
    }

    // Old specialized methods retained for now but unused
}

impl ExecContext for Interpreter {
    fn get_var(&self, name: &str) -> Option<&Argument> {
        self.variables.get(name)
    }

    fn set_var(&mut self, name: &str, value: Argument) {
        self.variables.insert(name.to_string(), value);
    }
}

fn m256i_to_argument(value: __m256i) -> Argument {
    let array: [i32; 8] = unsafe { std::mem::transmute(value) };
    let mut bytes = [0u8; 64];
    for (i, &val) in array.iter().enumerate() {
        let val_bytes = val.to_le_bytes();
        let start = i * 4;
        bytes[start..start + 4].copy_from_slice(&val_bytes);
    }
    Argument::Array(ArgType::I256, bytes)
}

fn display_argument_simple(arg: &Argument) {
    match arg {
        Argument::Scalar(val) => println!("{}", val),
        Argument::Array(arg_type, bytes) => match arg_type {
            ArgType::U8 => {
                let values_str: Vec<String> =
                    bytes.iter().take(32).map(|v| v.to_string()).collect();
                println!("b[{}]", values_str.join(", "));
            }
            ArgType::U16 => {
                let words: Vec<u16> = bytes
                    .chunks(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]))
                    .take(32)
                    .collect();
                let values_str: Vec<String> = words.iter().map(|v| v.to_string()).collect();
                println!("w[{}]", values_str.join(", "));
            }
            ArgType::U32 => {
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
            ArgType::U64 => {
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
            ArgType::I256 => {
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
                    .take(8)
                    .collect();
                let values_str: Vec<String> = dwords.iter().map(|v| v.to_string()).collect();
                println!("i256[{}]", values_str.join(", "));
            }
            ArgType::I512 => {
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
                println!("i512[{}]", values_str.join(", "));
            }
            ArgType::Ptr => {
                println!(
                    "ptr[0x{:016x}]",
                    u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7]
                    ])
                );
            }
        },
        Argument::Variable(name) => {
            println!("variable<{}>", name);
        }
        Argument::Memory(bytes) => println!("memory[{} bytes]", bytes.len()),
    }
}

fn valid_len_for_array(arg_type: &ArgType) -> usize {
    match arg_type {
        ArgType::I256 => 32,
        ArgType::I512 => 64,
        // Treat typed arrays as full 64 bytes
        ArgType::U8 | ArgType::U16 | ArgType::U32 | ArgType::U64 => 64,
        ArgType::Ptr => 8,
    }
}

fn print_hex(arg: &Argument, chunk_bits_opt: Option<u64>) -> Result<(), String> {
    let chunk_bits = chunk_bits_opt.unwrap_or(32);
    if !(chunk_bits == 1 || matches!(chunk_bits, 8 | 16 | 32 | 64)) {
        return Err("chunk_bits must be 1 (bitstring) or one of 8, 16, 32, 64".to_string());
    }
    match arg {
        Argument::Scalar(v) => {
            if chunk_bits == 1 {
                let bytes = v.to_le_bytes();
                let mut out = String::new();
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 { out.push('_'); }
                    out.push_str(&format!("{:08b}", b));
                }
                println!("{}", out);
            } else {
                match chunk_bits {
                    8 => println!("0x{:02x}", (v & 0xFF) as u8),
                    16 => println!("0x{:04x}", (v & 0xFFFF) as u16),
                    32 => println!("0x{:08x}", (v & 0xFFFF_FFFF) as u32),
                    64 => println!("0x{:016x}", *v as u64),
                    _ => unreachable!(),
                }
            }
        }
        Argument::Array(arg_type, bytes) => {
            let valid = valid_len_for_array(arg_type);
            let data = &bytes[..valid.min(bytes.len())];
            if chunk_bits == 1 {
                let mut out = String::new();
                for (i, b) in data.iter().enumerate() {
                    if i > 0 { out.push('_'); }
                    out.push_str(&format!("{:08b}", b));
                }
                println!("{}", out);
            } else {
                match chunk_bits {
                    8 => {
                        let mut out = String::new();
                        for b in data.iter() {
                            out.push_str(&format!("0x{:02x} ", b));
                        }
                        println!("{}", out.trim_end());
                    }
                    16 => {
                        for chunk in data.chunks_exact(2) {
                            let v = u16::from_le_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]);
                            print!("0x{:04x} ", v);
                        }
                        println!("");
                    }
                    32 => {
                        for chunk in data.chunks_exact(4) {
                            let v = u32::from_le_bytes([
                                chunk[0],
                                *chunk.get(1).unwrap_or(&0),
                                *chunk.get(2).unwrap_or(&0),
                                *chunk.get(3).unwrap_or(&0),
                            ]);
                            print!("0x{:08x} ", v);
                        }
                        println!("");
                    }
                    64 => {
                        for chunk in data.chunks_exact(8) {
                            let v = u64::from_le_bytes([
                                chunk[0],
                                *chunk.get(1).unwrap_or(&0),
                                *chunk.get(2).unwrap_or(&0),
                                *chunk.get(3).unwrap_or(&0),
                                *chunk.get(4).unwrap_or(&0),
                                *chunk.get(5).unwrap_or(&0),
                                *chunk.get(6).unwrap_or(&0),
                                *chunk.get(7).unwrap_or(&0),
                            ]);
                            print!("0x{:016x} ", v);
                        }
                        println!("");
                    }
                    _ => unreachable!(),
                }
            }
        }
        Argument::Variable(_) => unreachable!("variables are resolved before execute"),
        Argument::Memory(bytes) => {
            if chunk_bits == 1 {
                let mut out = String::new();
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 { out.push('_'); }
                    out.push_str(&format!("{:08b}", b));
                }
                println!("{}", out);
            } else {
                for b in bytes.iter() {
                    print!("0x{:02x} ", b);
                }
                println!("");
            }
        }
    }
    Ok(())
}

fn print_dec(arg: &Argument) -> Result<(), String> {
    match arg {
        Argument::Scalar(v) => {
            println!("{}", v);
        }
        Argument::Array(arg_type, bytes) => {
            let valid = valid_len_for_array(arg_type);
            let data = &bytes[..valid.min(bytes.len())];
            let chunk_bits = match arg_type {
                ArgType::U8 => 8,
                ArgType::U16 => 16,
                ArgType::U32 => 32,
                ArgType::U64 => 64,
                ArgType::I256 => 32,
                ArgType::I512 => 64,
                ArgType::Ptr => 64,
            };
            match chunk_bits {
                8 => {
                    for b in data.iter() {
                        print!("{} ", *b);
                    }
                    println!("");
                }
                16 => {
                    for chunk in data.chunks_exact(2) {
                        let v = u16::from_le_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]);
                        print!("{} ", v);
                    }
                    println!("");
                }
                32 => {
                    for chunk in data.chunks_exact(4) {
                        let v = u32::from_le_bytes([
                            chunk[0],
                            *chunk.get(1).unwrap_or(&0),
                            *chunk.get(2).unwrap_or(&0),
                            *chunk.get(3).unwrap_or(&0),
                        ]);
                        print!("{} ", v);
                    }
                    println!("");
                }
                64 => {
                    for chunk in data.chunks_exact(8) {
                        let v = u64::from_le_bytes([
                            chunk[0],
                            *chunk.get(1).unwrap_or(&0),
                            *chunk.get(2).unwrap_or(&0),
                            *chunk.get(3).unwrap_or(&0),
                            *chunk.get(4).unwrap_or(&0),
                            *chunk.get(5).unwrap_or(&0),
                            *chunk.get(6).unwrap_or(&0),
                            *chunk.get(7).unwrap_or(&0),
                        ]);
                        print!("{} ", v);
                    }
                    println!("");
                }
                _ => unreachable!(),
            }
        }
        Argument::Variable(_) => unreachable!("variables are resolved before execute"),
        Argument::Memory(bytes) => {
            for b in bytes.iter() {
                print!("{} ", *b);
            }
            println!("");
        }
    }
    Ok(())
}

fn print_bin(arg: &Argument) -> Result<(), String> {
    match arg {
        Argument::Scalar(v) => {
            println!("{:064b}", v);
        }
        Argument::Array(arg_type, bytes) => {
            let valid = valid_len_for_array(arg_type);
            let data = &bytes[..valid.min(bytes.len())];
            // Use natural chunking similar to decimal
            let chunk_bits = match arg_type {
                ArgType::U8 => 8,
                ArgType::U16 => 16,
                ArgType::U32 => 32,
                ArgType::U64 => 64,
                ArgType::I256 => 32,
                ArgType::I512 => 64,
                ArgType::Ptr => 64,
            };
            match chunk_bits {
                8 => {
                    for b in data.iter() {
                        print!("{:08b} ", b);
                    }
                    println!("");
                }
                16 => {
                    for chunk in data.chunks_exact(2) {
                        let v = u16::from_le_bytes([chunk[0], *chunk.get(1).unwrap_or(&0)]);
                        print!("{:016b} ", v);
                    }
                    println!("");
                }
                32 => {
                    for chunk in data.chunks_exact(4) {
                        let v = u32::from_le_bytes([
                            chunk[0],
                            *chunk.get(1).unwrap_or(&0),
                            *chunk.get(2).unwrap_or(&0),
                            *chunk.get(3).unwrap_or(&0),
                        ]);
                        print!("{:032b} ", v);
                    }
                    println!("");
                }
                64 => {
                    for chunk in data.chunks_exact(8) {
                        let v = u64::from_le_bytes([
                            chunk[0],
                            *chunk.get(1).unwrap_or(&0),
                            *chunk.get(2).unwrap_or(&0),
                            *chunk.get(3).unwrap_or(&0),
                            *chunk.get(4).unwrap_or(&0),
                            *chunk.get(5).unwrap_or(&0),
                            *chunk.get(6).unwrap_or(&0),
                            *chunk.get(7).unwrap_or(&0),
                        ]);
                        print!("{:064b} ", v);
                    }
                    println!("");
                }
                _ => unreachable!(),
            }
        }
        Argument::Variable(_) => unreachable!("variables are resolved before execute"),
        Argument::Memory(bytes) => {
            for b in bytes.iter() {
                print!("{:08b} ", b);
            }
            println!("");
        }
    }
    Ok(())
}
