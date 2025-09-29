use crate::ast::*;
use crate::avx_512::register_avx512_instructions;
use crate::avx2::register_avx2_instructions;
use crate::bmi2::register_bmi2_instructions;
use crate::sse2::register_sse2_instructions;
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

        registry.register_instruction(Instruction::with_arg_range(
            "test",
            vec![],
            ArgType::U64,
            0,
            Some(1),
            |_, args| {
                if args.len() > 1 {
                    return Err("test accepts at most 1 argument".to_string());
                }
                if let Some(first) = args.get(0) {
                    Ok(first.clone())
                } else {
                    Ok(Argument::Scalar(42))
                }
            },
        ));

        registry.register_instruction(Instruction::with_arg_range(
            "print",
            vec![],
            ArgType::U64,
            0,
            None,
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
        registry.register_instruction(Instruction::with_arg_range(
            "print_hex",
            vec![ArgType::U64, ArgType::U64],
            ArgType::U64,
            1,
            Some(2),
            |_, args| {
                if args.len() > 2 {
                    return Err("print_hex accepts at most 2 arguments".to_string());
                }
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
            vec![ArgType::U64],
            ArgType::U64,
            |_, args| {
                if args.len() != 1 {
                    return Err("print_dec requires exactly 1 argument".to_string());
                }
                print_dec(&args[0])?;
                Ok(Argument::Scalar(0))
            },
        ));

        // print_bin(value)
        registry.register_instruction(Instruction::new(
            "print_bin",
            vec![ArgType::U64],
            ArgType::U64,
            |_, args| {
                if args.len() != 1 {
                    return Err("print_bin requires exactly 1 argument".to_string());
                }
                print_bin(&args[0])?;
                Ok(Argument::Scalar(0))
            },
        ));

        registry.register_instruction(Instruction::with_arg_range(
            "print_str",
            vec![],
            ArgType::U64,
            0,
            Some(1),
            |ctx, args| {
                if args.len() > 1 {
                    return Err("print_str accepts at most 1 argument".to_string());
                }

                if args.is_empty() {
                    if let Some(res) = ctx.get_var("_res") {
                        let text = argument_to_utf8_lossy(res)?;
                        println!("{}", text);
                    } else {
                        println!("_res is undefined");
                    }
                } else {
                    let text = argument_to_utf8_lossy(&args[0])?;
                    println!("{}", text);
                }

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

        // Emulate memory-based expandload: _mm256_mask_expandloadu_epi32(src, k, mem)
        registry.register_instruction(Instruction::new(
            "_mm256_mask_expandloadu_epi32",
            vec![ArgType::I256, ArgType::U8, ArgType::Ptr],
            ArgType::I256,
            |ctx, args| {
                if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")) {
                    return Err(
                        "AVX-512F and AVX-512VL not supported on this CPU/runtime".to_string()
                    );
                }
                if args.len() != 3 {
                    return Err(
                        "_mm256_mask_expandloadu_epi32 requires exactly 3 arguments".to_string()
                    );
                }
                let src = args[0].to_i256();
                let mask = args[1].to_u8();

                // Resolve memory pointer
                let mem_ptr: *const i32 = match &args[2] {
                    Argument::Variable(name) => match ctx.get_var(name.as_str()) {
                        Some(Argument::Memory(bytes)) => bytes.as_ptr() as *const i32,
                        Some(other) => {
                            return Err(format!(
                                "Pointer '{}' does not reference memory, found {:?}",
                                name, other
                            ));
                        }
                        None => return Err(format!("Undefined pointer variable: {}", name)),
                    },
                    Argument::Memory(bytes) => bytes.as_ptr() as *const i32,
                    _ => {
                        return Err(
                            "Pointer argument must be a memory variable or mem[...]".to_string()
                        );
                    }
                };

                let result = unsafe { _mm256_mask_expandloadu_epi32(src, mask, mem_ptr) };
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

        // Register SSE2 intrinsic-backed instructions
        register_sse2_instructions(&mut registry);

        // Register AVX2 intrinsic-backed instructions
        register_avx2_instructions(&mut registry);
        // Register AVX-512 intrinsic-backed instructions
        register_avx512_instructions(&mut registry);
        // Register BMI2 intrinsic-backed instructions
        register_bmi2_instructions(&mut registry);

        Self {
            variables: HashMap::new(),
            function_registry: registry,
        }
    }

    pub fn execute(&mut self, ast: AST) -> Result<Argument, String> {
        ast.validate(&self.function_registry)?;
        let result = match ast {
            AST::Call { name, args } => self.execute_call(name, args),
            AST::Var { name, value } => {
                let resolved = self.resolve_argument(value.clone())?;
                self.variables.insert(name.clone(), resolved.clone());
                Ok(resolved)
            }
            AST::Assign { dest, child } => {
                let result = self.execute(*child)?;
                let resolved = self.resolve_argument(result)?;
                self.variables.insert(dest, resolved.clone());
                Ok(resolved)
            }
            AST::VarLookup { name } => self
                .variables
                .get(&name)
                .cloned()
                .ok_or_else(|| format!("Undefined variable: {}", name)),
            AST::MemoryStore {
                name,
                access,
                value,
            } => {
                let resolved_value = self.resolve_argument(value.clone())?;
                let data = argument_to_bytes(&resolved_value)?;
                let entry = self
                    .variables
                    .get_mut(&name)
                    .ok_or_else(|| format!("Undefined memory variable: {}", name))?;
                let bytes = match entry {
                    Argument::Memory(bytes) => bytes,
                    other => {
                        return Err(format!(
                            "Variable '{}' is not memory and cannot be indexed (found {:?})",
                            name, other
                        ));
                    }
                };
                let (start, end) = access.for_assignment(bytes.len(), data.len())?;
                bytes[start..end].copy_from_slice(&data);
                Ok(resolved_value)
            }
            AST::MemorySliceLookup { name, access } => {
                let data = self.read_memory_slice(&name, &access)?;
                Ok(Argument::Memory(data))
            }
        };

        // Store result in _res for all statements
        if let Ok(ref result) = result {
            self.variables.insert("_res".to_string(), result.clone());
        }

        result
    }

    fn execute_call(&mut self, name: String, args: Vec<Argument>) -> Result<Argument, String> {
        if let Some(func) = self.function_registry.find(&name) {
            let resolved_args: Vec<Argument> = if func.arguments().is_empty() {
                args.into_iter()
                    .map(|arg| self.resolve_argument(arg))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let sig = func.arguments();
                let mut out = Vec::with_capacity(args.len());
                for (idx, arg) in args.into_iter().enumerate() {
                    let keep_as_ptr = sig.get(idx).map(|t| *t == ArgType::Ptr).unwrap_or(false);
                    if keep_as_ptr {
                        if matches!(arg, Argument::MemorySlice { .. }) {
                            return Err("Pointer arguments do not currently support memory slices"
                                .to_string());
                        }
                        out.push(arg);
                    } else {
                        out.push(self.resolve_argument(arg)?);
                    }
                }
                out
            };

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
        Argument::ScalarTyped(_, val) => println!("{}", val),
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
            ArgType::I128 => {
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
                    .take(4)
                    .collect();
                let values_str: Vec<String> = dwords.iter().map(|v| v.to_string()).collect();
                println!("i128[{}]", values_str.join(", "));
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
        Argument::MemorySlice { name, .. } => println!("memory_slice<{}>", name),
    }
}

fn valid_len_for_array(arg_type: &ArgType) -> usize {
    match arg_type {
        ArgType::I128 => 16,
        ArgType::I256 => 32,
        ArgType::I512 => 64,
        // Treat typed arrays as full 64 bytes
        ArgType::U8 | ArgType::U16 | ArgType::U32 | ArgType::U64 => 64,
        ArgType::Ptr => 8,
    }
}

fn print_hex(arg: &Argument, chunk_bits_opt: Option<u64>) -> Result<(), String> {
    let chunk_bits = chunk_bits_opt.unwrap_or(32);
    if !(chunk_bits == 1 || matches!(chunk_bits, 8 | 16 | 32 | 64 | 128)) {
        return Err("chunk_bits must be 1 (bitstring) or one of 8, 16, 32, 64, 128".to_string());
    }
    match arg {
        Argument::Scalar(v) => {
            if chunk_bits == 1 {
                let bytes = v.to_le_bytes();
                let mut out = String::new();
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 {
                        out.push('_');
                    }
                    out.push_str(&format!("{:08b}", b));
                }
                println!("{}", out);
            } else {
                let bytes = v.to_le_bytes();
                match chunk_bits {
                    8 => {
                        let mut out = String::new();
                        for b in bytes.iter() {
                            out.push_str(&format!("0x{:02x} ", b));
                        }
                        println!("{}", out.trim_end());
                    }
                    16 => {
                        for i in (0..8).step_by(2) {
                            let vv = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
                            print!("0x{:04x} ", vv);
                        }
                        println!("");
                    }
                    32 => {
                        for i in (0..8).step_by(4) {
                            let vv = u32::from_le_bytes([
                                bytes[i],
                                bytes[i + 1],
                                bytes[i + 2],
                                bytes[i + 3],
                            ]);
                            print!("0x{:08x} ", vv);
                        }
                        println!("");
                    }
                    64 => {
                        println!("0x{:016x}", *v as u64);
                    }
                    _ => unreachable!(),
                }
            }
        }
        Argument::ScalarTyped(t, v) => {
            let type_bytes = t.byte_size();
            if chunk_bits == 1 {
                let mut bytes = v.to_le_bytes();
                // Zero out bytes beyond the declared type width for clarity
                for i in type_bytes..8 {
                    bytes[i] = 0;
                }
                let mut out = String::new();
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 {
                        out.push('_');
                    }
                    out.push_str(&format!("{:08b}", b));
                }
                println!("{}", out);
            } else {
                let bytes = v.to_le_bytes();
                match chunk_bits {
                    8 => {
                        let mut out = String::new();
                        for i in 0..type_bytes {
                            out.push_str(&format!("0x{:02x} ", bytes[i]));
                        }
                        println!("{}", out.trim_end());
                    }
                    16 => {
                        let total = (type_bytes + 1) / 2;
                        for n in 0..total {
                            let i = n * 2;
                            let vv = u16::from_le_bytes([
                                bytes.get(i).copied().unwrap_or(0),
                                bytes.get(i + 1).copied().unwrap_or(0),
                            ]);
                            print!("0x{:04x} ", vv);
                        }
                        println!("");
                    }
                    32 => {
                        let total = (type_bytes + 3) / 4;
                        for n in 0..total {
                            let i = n * 4;
                            let vv = u32::from_le_bytes([
                                bytes.get(i).copied().unwrap_or(0),
                                bytes.get(i + 1).copied().unwrap_or(0),
                                bytes.get(i + 2).copied().unwrap_or(0),
                                bytes.get(i + 3).copied().unwrap_or(0),
                            ]);
                            print!("0x{:08x} ", vv);
                        }
                        println!("");
                    }
                    64 => {
                        let vv = if type_bytes >= 8 {
                            *v as u64
                        } else {
                            let mut tmp = [0u8; 8];
                            tmp[..type_bytes].copy_from_slice(&bytes[..type_bytes]);
                            u64::from_le_bytes(tmp)
                        };
                        println!("0x{:016x}", vv);
                    }
                    _ => unreachable!(),
                }
            }
        }
        Argument::Array(arg_type, bytes) => {
            let valid = arg_type.vector_byte_len();
            let data = &bytes[..valid.min(bytes.len())];
            if chunk_bits == 1 {
                let mut out = String::new();
                for (i, b) in data.iter().enumerate() {
                    if i > 0 {
                        out.push('_');
                    }
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
                    128 => {
                        for chunk in data.chunks_exact(16) {
                            let v_low = u64::from_le_bytes([
                                chunk[0],
                                *chunk.get(1).unwrap_or(&0),
                                *chunk.get(2).unwrap_or(&0),
                                *chunk.get(3).unwrap_or(&0),
                                *chunk.get(4).unwrap_or(&0),
                                *chunk.get(5).unwrap_or(&0),
                                *chunk.get(6).unwrap_or(&0),
                                *chunk.get(7).unwrap_or(&0),
                            ]);
                            let v_high = u64::from_le_bytes([
                                *chunk.get(8).unwrap_or(&0),
                                *chunk.get(9).unwrap_or(&0),
                                *chunk.get(10).unwrap_or(&0),
                                *chunk.get(11).unwrap_or(&0),
                                *chunk.get(12).unwrap_or(&0),
                                *chunk.get(13).unwrap_or(&0),
                                *chunk.get(14).unwrap_or(&0),
                                *chunk.get(15).unwrap_or(&0),
                            ]);
                            print!("0x{:016x}{:016x} ", v_high, v_low);
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
                    if i > 0 {
                        out.push('_');
                    }
                    out.push_str(&format!("{:08b}", b));
                }
                println!("{}", out);
            } else {
                print_memory_hex(bytes, chunk_bits as usize);
            }
        }
        Argument::MemorySlice { .. } => unreachable!("memory slices are resolved before printing"),
    }
    Ok(())
}

fn print_dec(arg: &Argument) -> Result<(), String> {
    match arg {
        Argument::Scalar(v) => {
            println!("{}", v);
        }
        Argument::ScalarTyped(_, v) => {
            println!("{}", v);
        }
        Argument::Array(arg_type, bytes) => {
            let valid = arg_type.vector_byte_len();
            let data = &bytes[..valid.min(bytes.len())];
            let chunk_bits = match arg_type {
                ArgType::U8 => 8,
                ArgType::U16 => 16,
                ArgType::U32 => 32,
                ArgType::U64 => 64,
                ArgType::I128 => 32,
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
        Argument::MemorySlice { .. } => unreachable!("memory slices are resolved before printing"),
    }
    Ok(())
}

fn print_bin(arg: &Argument) -> Result<(), String> {
    match arg {
        Argument::Scalar(v) => {
            println!("{:064b}", v);
        }
        Argument::ScalarTyped(_, v) => {
            println!("{:064b}", v);
        }
        Argument::Array(arg_type, bytes) => {
            let valid = arg_type.vector_byte_len();
            let data = &bytes[..valid.min(bytes.len())];
            // Use natural chunking similar to decimal
            let chunk_bits = match arg_type {
                ArgType::U8 => 8,
                ArgType::U16 => 16,
                ArgType::U32 => 32,
                ArgType::U64 => 64,
                ArgType::I128 => 32,
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
        Argument::MemorySlice { .. } => unreachable!("memory slices are resolved before printing"),
    }
    Ok(())
}

pub(crate) fn argument_to_utf8_lossy(arg: &Argument) -> Result<String, String> {
    match arg {
        Argument::Scalar(val) => {
            let bytes = val.to_le_bytes();
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
        }
        Argument::ScalarTyped(arg_type, val) => {
            let bytes = val.to_le_bytes();
            let len = arg_type.byte_size().min(bytes.len());
            let slice = &bytes[..len];
            let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
            Ok(String::from_utf8_lossy(&slice[..end]).into_owned())
        }
        Argument::Array(arg_type, bytes) => {
            let len = arg_type.vector_byte_len().min(bytes.len());
            let slice = &bytes[..len];
            let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
            Ok(String::from_utf8_lossy(&slice[..end]).into_owned())
        }
        Argument::Memory(bytes) => {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
        }
        Argument::Variable(name) => {
            Err(format!("print_str received unresolved variable '{}'", name))
        }
        Argument::MemorySlice { name, .. } => Err(format!(
            "print_str received unresolved memory slice '{}[...]'",
            name
        )),
    }
}

impl Interpreter {
    fn resolve_argument(&self, arg: Argument) -> Result<Argument, String> {
        match arg {
            Argument::Variable(name) => {
                let value = self
                    .variables
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", name))?;
                self.resolve_argument(value)
            }
            Argument::MemorySlice { name, access } => {
                let data = self.read_memory_slice(&name, &access)?;
                Ok(Argument::Memory(data))
            }
            Argument::Memory(bytes) => Ok(Argument::Memory(bytes)),
            Argument::Array(arg_type, bytes) => Ok(Argument::Array(arg_type, bytes)),
            Argument::Scalar(v) => Ok(Argument::Scalar(v)),
            Argument::ScalarTyped(t, v) => Ok(Argument::ScalarTyped(t, v)),
        }
    }

    fn read_memory_slice(&self, name: &str, access: &MemoryAccess) -> Result<Vec<u8>, String> {
        let entry = self
            .variables
            .get(name)
            .ok_or_else(|| format!("Undefined memory variable: {}", name))?;
        let bytes = match entry {
            Argument::Memory(bytes) => bytes,
            other => {
                return Err(format!(
                    "Variable '{}' is not memory and cannot be sliced (found {:?})",
                    name, other
                ));
            }
        };
        let (start, end) = access.for_lookup(bytes.len())?;
        Ok(bytes[start..end].to_vec())
    }
}

fn argument_to_bytes(arg: &Argument) -> Result<Vec<u8>, String> {
    match arg {
        Argument::Memory(bytes) => Ok(bytes.clone()),
        Argument::Array(arg_type, bytes) => {
            let len = arg_type.vector_byte_len().min(bytes.len());
            Ok(bytes[..len].to_vec())
        }
        Argument::Scalar(v) => Ok(vec![(*v & 0xFF) as u8]),
        Argument::ScalarTyped(arg_type, v) => match arg_type {
            ArgType::U8 | ArgType::U16 | ArgType::U32 | ArgType::U64 => {
                let width = arg_type.byte_size();
                let le = v.to_le_bytes();
                Ok(le[..width].to_vec())
            }
            _ => {
                Err("Only u8/u16/u32/u64 typed scalars are supported for memory writes".to_string())
            }
        },
        Argument::Variable(name) => Err(format!(
            "Unresolved variable '{}' encountered during memory write",
            name
        )),
        Argument::MemorySlice { .. } => {
            Err("Memory slice must be resolved before writing".to_string())
        }
    }
}

fn print_memory_hex(bytes: &[u8], chunk_bits: usize) {
    match chunk_bits {
        8 => {
            for b in bytes {
                print!("0x{:02x} ", b);
            }
            println!("");
        }
        16 => {
            for chunk in bytes.chunks(2) {
                let v = u16::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                ]);
                print!("0x{:04x} ", v);
            }
            println!("");
        }
        32 => {
            for chunk in bytes.chunks(4) {
                let v = u32::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                print!("0x{:08x} ", v);
            }
            println!("");
        }
        64 => {
            for chunk in bytes.chunks(8) {
                let v = u64::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                    chunk.get(4).copied().unwrap_or(0),
                    chunk.get(5).copied().unwrap_or(0),
                    chunk.get(6).copied().unwrap_or(0),
                    chunk.get(7).copied().unwrap_or(0),
                ]);
                print!("0x{:016x} ", v);
            }
            println!("");
        }
        128 => {
            for chunk in bytes.chunks(16) {
                let low = u64::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                    chunk.get(4).copied().unwrap_or(0),
                    chunk.get(5).copied().unwrap_or(0),
                    chunk.get(6).copied().unwrap_or(0),
                    chunk.get(7).copied().unwrap_or(0),
                ]);
                let high = u64::from_le_bytes([
                    chunk.get(8).copied().unwrap_or(0),
                    chunk.get(9).copied().unwrap_or(0),
                    chunk.get(10).copied().unwrap_or(0),
                    chunk.get(11).copied().unwrap_or(0),
                    chunk.get(12).copied().unwrap_or(0),
                    chunk.get(13).copied().unwrap_or(0),
                    chunk.get(14).copied().unwrap_or(0),
                    chunk.get(15).copied().unwrap_or(0),
                ]);
                print!("0x{:016x}{:016x} ", high, low);
            }
            println!("");
        }
        _ => {}
    }
}
