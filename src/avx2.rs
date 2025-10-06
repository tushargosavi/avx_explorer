use crate::ast::{ArgType, Argument, ExecContext, FunctionRegistry, Instruction};
use std::arch::x86_64::*;
use std::convert::TryFrom;

fn m128i_to_argument(value: __m128i) -> Argument {
    let array: [i32; 4] = unsafe { std::mem::transmute(value) };
    let mut bytes = [0u8; 64];
    for (i, &val) in array.iter().enumerate() {
        let val_bytes = val.to_le_bytes();
        let start = i * 4;
        bytes[start..start + 4].copy_from_slice(&val_bytes);
    }
    Argument::Array(ArgType::I128, bytes)
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

fn clone_memory_from_ptr(
    ctx: &mut dyn ExecContext,
    ptr_arg: &Argument,
    instr: &str,
) -> Result<Vec<u8>, String> {
    match ptr_arg {
        Argument::Variable(name) => match ctx.get_var(name) {
            Some(Argument::Memory(bytes)) => Ok(bytes.clone()),
            Some(other) => Err(format!(
                "{} pointer '{}' does not reference memory (found {:?})",
                instr, name, other
            )),
            None => Err(format!("Undefined pointer variable: {}", name)),
        },
        Argument::Memory(bytes) => Ok(bytes.clone()),
        _ => Err(format!("{} first argument must be a memory pointer", instr)),
    }
}

fn extract_offset(arg: Option<&Argument>) -> Result<usize, String> {
    if let Some(arg) = arg {
        usize::try_from(arg.to_u64())
            .map_err(|_| "Offset exceeds addressable range on this platform".to_string())
    } else {
        Ok(0)
    }
}

fn m256i_to_bytes(value: __m256i) -> [u8; 32] {
    let lanes: [i32; 8] = unsafe { std::mem::transmute(value) };
    let mut bytes = [0u8; 32];
    for (idx, lane) in lanes.iter().enumerate() {
        let le = lane.to_le_bytes();
        let start = idx * 4;
        bytes[start..start + 4].copy_from_slice(&le);
    }
    bytes
}

fn bytes_to_m256i(bytes: &[u8]) -> __m256i {
    let mut lanes = [0i32; 8];
    for i in 0..8 {
        let start = i * 4;
        lanes[i] = i32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ]);
    }
    unsafe { std::mem::transmute(lanes) }
}

fn load_vector_bytes<const N: usize>(
    memory: &[u8],
    offset: usize,
    instr: &str,
) -> Result<[u8; N], String> {
    if offset > memory.len() {
        return Err(format!(
            "{} reading {} byte(s) at offset {} exceeds memory length {}",
            instr,
            N,
            offset,
            memory.len()
        ));
    }

    let mut buffer = [0u8; N];
    let available = memory.len() - offset;
    let to_copy = available.min(N);
    if to_copy > 0 {
        buffer[..to_copy].copy_from_slice(&memory[offset..offset + to_copy]);
    }
    Ok(buffer)
}

fn ensure_alignment(offset: usize, alignment: usize, instr: &str) -> Result<(), String> {
    if offset % alignment != 0 {
        Err(format!(
            "{} requires a {}-byte aligned offset (got {})",
            instr, alignment, offset
        ))
    } else {
        Ok(())
    }
}

#[inline]
fn require_avx() -> Result<(), String> {
    if !is_x86_feature_detected!("avx") {
        Err("AVX not supported on this CPU/runtime".to_string())
    } else {
        Ok(())
    }
}

#[inline]
fn require_avx2() -> Result<(), String> {
    if !is_x86_feature_detected!("avx2") {
        Err("AVX2 not supported on this CPU/runtime".to_string())
    } else {
        Ok(())
    }
}

pub fn register_avx2_instructions(registry: &mut FunctionRegistry) {
    // set1/broadcast
    registry.register_instruction(Instruction::new(
        "_mm256_set1_epi8",
        vec![ArgType::U8],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_set1_epi8 requires 1 argument".to_string());
            }
            let v = args[0].to_u8() as i8;
            let res = unsafe { _mm256_set1_epi8(v) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_set1_epi16",
        vec![ArgType::U16],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_set1_epi16 requires 1 argument".to_string());
            }
            let v = args[0].to_u16() as i16;
            let res = unsafe { _mm256_set1_epi16(v) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_set1_epi32",
        vec![ArgType::U32],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_set1_epi32 requires 1 argument".to_string());
            }
            let v = args[0].to_u32() as i32;
            let res = unsafe { _mm256_set1_epi32(v) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_set1_epi64x",
        vec![ArgType::U64],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_set1_epi64x requires 1 argument".to_string());
            }
            let v = args[0].to_u64() as i64;
            let res = unsafe { _mm256_set1_epi64x(v) };
            Ok(m256i_to_argument(res))
        },
    ));

    // AVX intrinsics
    registry.register_instruction(Instruction::new(
        "_mm_set1_epi8",
        vec![ArgType::U8],
        ArgType::I128,
        |_, args| {
            require_avx()?;
            if args.len() != 1 {
                return Err("_mm_set1_epi8 requires 1 argument".to_string());
            }
            let v = args[0].to_u8() as i8;
            let res = unsafe { _mm_set1_epi8(v) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cvtsi128_si64",
        vec![ArgType::I128],
        ArgType::U64,
        |_, args| {
            require_avx()?;
            if args.len() != 1 {
                return Err("_mm_cvtsi128_si64 requires 1 argument".to_string());
            }
            let a = args[0].to_i128();
            let res = unsafe { _mm_cvtsi128_si64(a) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U64, res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_set1_epi64x",
        vec![ArgType::U64],
        ArgType::I128,
        |_, args| {
            require_avx()?;
            if args.len() != 1 {
                return Err("_mm_set1_epi64x requires 1 argument".to_string());
            }
            let v = args[0].to_u64() as i64;
            let res = unsafe { _mm_set1_epi64x(v) };
            Ok(m128i_to_argument(res))
        },
    ));

    // add
    registry.register_instruction(Instruction::new(
        "_mm256_add_epi8",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_add_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_add_epi8(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_add_epi16",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_add_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_add_epi16(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_add_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_add_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_add_epi32(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_add_epi64",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_add_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_add_epi64(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));

    // sub
    registry.register_instruction(Instruction::new(
        "_mm256_sub_epi8",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sub_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_sub_epi8(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sub_epi16",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sub_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_sub_epi16(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sub_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sub_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_sub_epi32(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sub_epi64",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sub_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_sub_epi64(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));

    // mullo
    registry.register_instruction(Instruction::new(
        "_mm256_mullo_epi16",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_mullo_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_mullo_epi16(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_mullo_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_mullo_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_mullo_epi32(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));

    // bitwise
    registry.register_instruction(Instruction::new(
        "_mm256_and_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_and_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_and_si256(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_or_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_or_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_or_si256(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_xor_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_xor_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_xor_si256(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_andnot_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_andnot_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_andnot_si256(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));

    // compares
    registry.register_instruction(Instruction::new(
        "_mm256_cmpeq_epi8",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpeq_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpeq_epi8(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_cmpeq_epi16",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpeq_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpeq_epi16(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_cmpeq_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpeq_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpeq_epi32(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_cmpeq_epi64",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpeq_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpeq_epi64(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm256_cmpgt_epi8",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpgt_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpgt_epi8(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_cmpgt_epi16",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpgt_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpgt_epi16(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_cmpgt_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpgt_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpgt_epi32(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_cmpgt_epi64",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_cmpgt_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_cmpgt_epi64(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));

    // abs
    registry.register_instruction(Instruction::new(
        "_mm256_abs_epi8",
        vec![ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_abs_epi8 requires 1 argument".to_string());
            }
            let a = args[0].to_i256();
            let res = unsafe { _mm256_abs_epi8(a) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_abs_epi16",
        vec![ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_abs_epi16 requires 1 argument".to_string());
            }
            let a = args[0].to_i256();
            let res = unsafe { _mm256_abs_epi16(a) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_abs_epi32",
        vec![ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_abs_epi32 requires 1 argument".to_string());
            }
            let a = args[0].to_i256();
            let res = unsafe { _mm256_abs_epi32(a) };
            Ok(m256i_to_argument(res))
        },
    ));

    // shifts with shared count vectors
    registry.register_instruction(Instruction::new(
        "_mm256_sll_epi16",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sll_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sll_epi16(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "__mm256_sll_epi16",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("__mm256_sll_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sll_epi16(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sll_epi32",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sll_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sll_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "__mm256_sll_epi32",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("__mm256_sll_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sll_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sll_epi64",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sll_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sll_epi64(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "__mm256_sll_epi64",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("__mm256_sll_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sll_epi64(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_srl_epi16",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_srl_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_srl_epi16(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "__mm256_srl_epi16",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("__mm256_srl_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_srl_epi16(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_srl_epi32",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_srl_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_srl_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "__mm256_srl_epi32",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("__mm256_srl_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_srl_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_srl_epi64",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_srl_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_srl_epi64(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "__mm256_srl_epi64",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("__mm256_srl_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_srl_epi64(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sra_epi16",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sra_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sra_epi16(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sra_epi32",
        vec![ArgType::I256, ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sra_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i128();
            let res = unsafe { _mm256_sra_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));

    // variable shifts (no const immediates)
    registry.register_instruction(Instruction::new(
        "_mm256_sllv_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sllv_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i256();
            let res = unsafe { _mm256_sllv_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_sllv_epi64",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_sllv_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i256();
            let res = unsafe { _mm256_sllv_epi64(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_srlv_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_srlv_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i256();
            let res = unsafe { _mm256_srlv_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_srlv_epi64",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_srlv_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i256();
            let res = unsafe { _mm256_srlv_epi64(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_srav_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_srav_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let count = args[1].to_i256();
            let res = unsafe { _mm256_srav_epi32(a, count) };
            Ok(m256i_to_argument(res))
        },
    ));

    // mask extraction and tests
    registry.register_instruction(Instruction::new(
        "_mm256_movemask_epi8",
        vec![ArgType::I256],
        ArgType::U32,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_movemask_epi8 requires 1 argument".to_string());
            }
            let a = args[0].to_i256();
            let res = unsafe { _mm256_movemask_epi8(a) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U32, res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_testz_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::U32,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_testz_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_testz_si256(a, b) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U32, res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_testc_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::U32,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_testc_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_testc_si256(a, b) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U32, res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_testnzc_si256",
        vec![ArgType::I256, ArgType::I256],
        ArgType::U32,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_testnzc_si256 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_testnzc_si256(a, b) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U32, res))
        },
    ));

    // blend/shuffle/permute (non-imm variants)
    registry.register_instruction(Instruction::new(
        "_mm256_blendv_epi8",
        vec![ArgType::I256, ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 3 {
                return Err("_mm256_blendv_epi8 requires 3 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let mask = args[2].to_i256();
            let res = unsafe { _mm256_blendv_epi8(a, b, mask) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_shuffle_epi8",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_shuffle_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let b = args[1].to_i256();
            let res = unsafe { _mm256_shuffle_epi8(a, b) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_permutevar8x32_epi32",
        vec![ArgType::I256, ArgType::I256],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_permutevar8x32_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i256();
            let idx = args[1].to_i256();
            let res = unsafe { _mm256_permutevar8x32_epi32(a, idx) };
            Ok(m256i_to_argument(res))
        },
    ));

    // AVX2 load/store operations
    registry.register_instruction(Instruction::with_arg_range(
        "_mm256_load_si256",
        vec![ArgType::Ptr, ArgType::U64],
        ArgType::I256,
        1,
        Some(2),
        |ctx, args| {
            require_avx()?;
            if args.len() < 1 || args.len() > 2 {
                return Err("_mm256_load_si256 expects 1 or 2 arguments".to_string());
            }

            let offset = extract_offset(args.get(1))?;
            ensure_alignment(offset, 32, "_mm256_load_si256")?;

            let memory = clone_memory_from_ptr(ctx, &args[0], "_mm256_load_si256")?;
            let bytes = load_vector_bytes::<32>(&memory, offset, "_mm256_load_si256")?;
            let vec = bytes_to_m256i(&bytes);
            Ok(m256i_to_argument(vec))
        },
    ));

    registry.register_instruction(Instruction::with_arg_range(
        "_mm256_loadu_si256",
        vec![ArgType::Ptr, ArgType::U64],
        ArgType::I256,
        1,
        Some(2),
        |ctx, args| {
            require_avx()?;
            if args.len() < 1 || args.len() > 2 {
                return Err("_mm256_loadu_si256 expects 1 or 2 arguments".to_string());
            }

            let offset = extract_offset(args.get(1))?;
            let memory = clone_memory_from_ptr(ctx, &args[0], "_mm256_loadu_si256")?;
            let bytes = load_vector_bytes::<32>(&memory, offset, "_mm256_loadu_si256")?;
            let vec = bytes_to_m256i(&bytes);
            Ok(m256i_to_argument(vec))
        },
    ));

    registry.register_instruction(Instruction::with_arg_range(
        "_mm256_store_si256",
        vec![ArgType::Ptr, ArgType::I256, ArgType::U64],
        ArgType::Ptr,
        2,
        Some(3),
        |ctx, args| {
            require_avx()?;
            if args.len() < 2 || args.len() > 3 {
                return Err("_mm256_store_si256 expects 2 or 3 arguments".to_string());
            }

            let mut memory = clone_memory_from_ptr(ctx, &args[0], "_mm256_store_si256")?;
            let offset = extract_offset(args.get(2))?;
            ensure_alignment(offset, 32, "_mm256_store_si256")?;

            let value = args[1].to_i256();
            let bytes = m256i_to_bytes(value);
            let end = offset
                .checked_add(bytes.len())
                .ok_or_else(|| "Offset calculation overflowed".to_string())?;
            if memory.len() < end {
                memory.resize(end, 0);
            }
            memory[offset..end].copy_from_slice(&bytes);

            Ok(Argument::Memory(memory))
        },
    ));

    registry.register_instruction(Instruction::with_arg_range(
        "_mm256_storeu_si256",
        vec![ArgType::Ptr, ArgType::I256, ArgType::U64],
        ArgType::Ptr,
        2,
        Some(3),
        |ctx, args| {
            require_avx()?;
            if args.len() < 2 || args.len() > 3 {
                return Err("_mm256_storeu_si256 expects 2 or 3 arguments".to_string());
            }

            let mut memory = clone_memory_from_ptr(ctx, &args[0], "_mm256_storeu_si256")?;
            let offset = extract_offset(args.get(2))?;

            let value = args[1].to_i256();
            let bytes = m256i_to_bytes(value);
            let end = offset
                .checked_add(bytes.len())
                .ok_or_else(|| "Offset calculation overflowed".to_string())?;
            if memory.len() < end {
                memory.resize(end, 0);
            }
            memory[offset..end].copy_from_slice(&bytes);

            Ok(Argument::Memory(memory))
        },
    ));

    // AVX2 conversion and store operations
    registry.register_instruction(Instruction::new(
        "_mm256_cvtepi8_epi32",
        vec![ArgType::I128],
        ArgType::I256,
        |_, args| {
            require_avx2()?;
            if args.len() != 1 {
                return Err("_mm256_cvtepi8_epi32 requires 1 argument".to_string());
            }
            let a = args[0].to_i128();
            let res = unsafe { _mm256_cvtepi8_epi32(a) };
            Ok(m256i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm256_storeu_epi32",
        vec![ArgType::Ptr, ArgType::I256],
        ArgType::Ptr,
        |ctx, args| {
            require_avx2()?;
            if args.len() != 2 {
                return Err("_mm256_storeu_epi32 requires 2 arguments".to_string());
            }

            // Get the memory to store to
            let mem_arg = match &args[0] {
                Argument::Variable(name) => ctx
                    .get_var(name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))?,
                Argument::Memory(_) => &args[0],
                _ => {
                    return Err(
                        "_mm256_storeu_epi32 first argument must be a memory pointer".to_string(),
                    );
                }
            };

            let memory = match mem_arg {
                Argument::Memory(bytes) => bytes.clone(),
                _ => {
                    return Err(
                        "_mm256_storeu_epi32 first argument must reference memory".to_string()
                    );
                }
            };

            let value = args[1].to_i256();
            let value_bytes: [i32; 8] = unsafe { std::mem::transmute(value) };

            // Ensure memory is large enough (32 bytes for 8 i32s)
            let mut new_memory = memory;
            if new_memory.len() < 32 {
                new_memory.resize(32, 0);
            }

            // Store the 8 i32 values (little endian)
            for i in 0..8 {
                let offset = i * 4;
                let bytes = value_bytes[i].to_le_bytes();
                new_memory[offset..offset + 4].copy_from_slice(&bytes);
            }

            Ok(Argument::Memory(new_memory))
        },
    ));
}
