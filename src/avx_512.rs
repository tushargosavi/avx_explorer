use crate::ast::{ArgType, Argument, ExecContext, FunctionRegistry, Instruction};
use std::arch::x86_64::*;
use std::convert::TryFrom;

fn m512i_to_argument(value: __m512i) -> Argument {
    let lanes: [i64; 8] = unsafe { std::mem::transmute(value) };
    let mut bytes = [0u8; 64];
    for (i, &lane) in lanes.iter().enumerate() {
        let lane_bytes = lane.to_le_bytes();
        let start = i * 8;
        bytes[start..start + 8].copy_from_slice(&lane_bytes);
    }
    Argument::Array(ArgType::I512, bytes)
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

fn m512i_to_bytes(value: __m512i) -> [u8; 64] {
    let lanes: [i64; 8] = unsafe { std::mem::transmute(value) };
    let mut bytes = [0u8; 64];
    for (idx, lane) in lanes.iter().enumerate() {
        let le = lane.to_le_bytes();
        let start = idx * 8;
        bytes[start..start + 8].copy_from_slice(&le);
    }
    bytes
}

fn bytes_to_m512i(bytes: &[u8]) -> __m512i {
    let mut lanes = [0i64; 8];
    for i in 0..8 {
        let start = i * 8;
        lanes[i] = i64::from_le_bytes([
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
fn require_avx512f() -> Result<(), String> {
    if !is_x86_feature_detected!("avx512f") {
        Err("AVX-512F not supported on this CPU/runtime".to_string())
    } else {
        Ok(())
    }
}

#[inline]
fn require_avx512dq() -> Result<(), String> {
    if !is_x86_feature_detected!("avx512dq") {
        Err("AVX-512DQ not supported on this CPU/runtime".to_string())
    } else {
        Ok(())
    }
}

pub fn register_avx512_instructions(registry: &mut FunctionRegistry) {
    // set1/broadcast
    registry.register_instruction(Instruction::new(
        "_mm512_set1_epi32",
        vec![ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 1 {
                return Err("_mm512_set1_epi32 requires 1 argument".to_string());
            }
            let v = args[0].to_u32() as i32;
            let res = unsafe { _mm512_set1_epi32(v) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_set1_epi64",
        vec![ArgType::U64],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 1 {
                return Err("_mm512_set1_epi64 requires 1 argument".to_string());
            }
            let v = args[0].to_u64() as i64;
            let res = unsafe { _mm512_set1_epi64(v) };
            Ok(m512i_to_argument(res))
        },
    ));

    // AVX-512 load/store operations
    registry.register_instruction(Instruction::with_arg_range(
        "_mm512_load_si512",
        vec![ArgType::Ptr, ArgType::U64],
        ArgType::I512,
        1,
        Some(2),
        |ctx, args| {
            if args.len() < 1 || args.len() > 2 {
                return Err("_mm512_load_si512 expects 1 or 2 arguments".to_string());
            }

            let offset = extract_offset(args.get(1))?;
            ensure_alignment(offset, 64, "_mm512_load_si512")?;

            let memory = clone_memory_from_ptr(ctx, &args[0], "_mm512_load_si512")?;
            let bytes = load_vector_bytes::<64>(&memory, offset, "_mm512_load_si512")?;
            let vec = bytes_to_m512i(&bytes);
            Ok(m512i_to_argument(vec))
        },
    ));

    registry.register_instruction(Instruction::with_arg_range(
        "_mm512_loadu_si512",
        vec![ArgType::Ptr, ArgType::U64],
        ArgType::I512,
        1,
        Some(2),
        |ctx, args| {
            if args.len() < 1 || args.len() > 2 {
                return Err("_mm512_loadu_si512 expects 1 or 2 arguments".to_string());
            }

            let offset = extract_offset(args.get(1))?;
            let memory = clone_memory_from_ptr(ctx, &args[0], "_mm512_loadu_si512")?;
            let bytes = load_vector_bytes::<64>(&memory, offset, "_mm512_loadu_si512")?;
            let vec = bytes_to_m512i(&bytes);
            Ok(m512i_to_argument(vec))
        },
    ));

    registry.register_instruction(Instruction::with_arg_range(
        "_mm512_store_si512",
        vec![ArgType::Ptr, ArgType::I512, ArgType::U64],
        ArgType::Ptr,
        2,
        Some(3),
        |ctx, args| {
            if args.len() < 2 || args.len() > 3 {
                return Err("_mm512_store_si512 expects 2 or 3 arguments".to_string());
            }

            let mut memory = clone_memory_from_ptr(ctx, &args[0], "_mm512_store_si512")?;
            let offset = extract_offset(args.get(2))?;
            ensure_alignment(offset, 64, "_mm512_store_si512")?;

            let value = args[1].to_i512();
            let bytes = m512i_to_bytes(value);
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
        "_mm512_storeu_si512",
        vec![ArgType::Ptr, ArgType::I512, ArgType::U64],
        ArgType::Ptr,
        2,
        Some(3),
        |ctx, args| {
            if args.len() < 2 || args.len() > 3 {
                return Err("_mm512_storeu_si512 expects 2 or 3 arguments".to_string());
            }

            let mut memory = clone_memory_from_ptr(ctx, &args[0], "_mm512_storeu_si512")?;
            let offset = extract_offset(args.get(2))?;

            let value = args[1].to_i512();
            let bytes = m512i_to_bytes(value);
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

    // add
    registry.register_instruction(Instruction::new(
        "_mm512_add_epi32",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_add_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_add_epi32(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_add_epi64",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_add_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_add_epi64(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    // sub
    registry.register_instruction(Instruction::new(
        "_mm512_sub_epi32",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_sub_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_sub_epi32(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_sub_epi64",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_sub_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_sub_epi64(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    // mullo (32-bit)
    registry.register_instruction(Instruction::new(
        "_mm512_mullo_epi32",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_mullo_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_mullo_epi32(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    // logical operations
    registry.register_instruction(Instruction::new(
        "_mm512_and_si512",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_and_si512 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_and_si512(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_or_si512",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_or_si512 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_or_si512(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_xor_si512",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_xor_si512 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let res = unsafe { _mm512_xor_si512(a, b) };
            Ok(m512i_to_argument(res))
        },
    ));

    // shift immediate operations
    registry.register_instruction(Instruction::new(
        "_mm512_slli_epi32",
        vec![ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_slli_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let imm = (args[1].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_sllv_epi32(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_sll_epi32",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_sll_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_sll_epi32(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "__mm512_sll_epi32",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("__mm512_sll_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_sll_epi32(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_slli_epi64",
        vec![ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_slli_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let imm = (args[1].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_sllv_epi64(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_sll_epi64",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_sll_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_sll_epi64(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "__mm512_sll_epi64",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("__mm512_sll_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_sll_epi64(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srli_epi32",
        vec![ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_srli_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let imm = (args[1].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_srlv_epi32(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srl_epi32",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_srl_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_srl_epi32(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "__mm512_srl_epi32",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("__mm512_srl_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_srl_epi32(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srli_epi64",
        vec![ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_srli_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let imm = (args[1].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_srlv_epi64(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srl_epi64",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_srl_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_srl_epi64(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "__mm512_srl_epi64",
        vec![ArgType::I512, ArgType::I128],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("__mm512_srl_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let count = args[1].to_i128();
            let res = unsafe { _mm512_srl_epi64(a, count) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srai_epi32",
        vec![ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_srai_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let imm = (args[1].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_srav_epi32(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srai_epi64",
        vec![ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_srai_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let imm = (args[1].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_srav_epi64(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    // shift immediate mask operations
    registry.register_instruction(Instruction::new(
        "_mm512_mask_slli_epi32",
        vec![ArgType::I512, ArgType::U16, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 4 {
                return Err("_mm512_mask_slli_epi32 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u16();
            let a = args[2].to_i512();
            let imm = (args[3].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_mask_sllv_epi32(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_slli_epi64",
        vec![ArgType::I512, ArgType::U8, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 4 {
                return Err("_mm512_mask_slli_epi64 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u8();
            let a = args[2].to_i512();
            let imm = (args[3].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_mask_sllv_epi64(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srli_epi32",
        vec![ArgType::I512, ArgType::U16, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srli_epi32 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u16();
            let a = args[2].to_i512();
            let imm = (args[3].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_mask_srlv_epi32(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srli_epi64",
        vec![ArgType::I512, ArgType::U8, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srli_epi64 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u8();
            let a = args[2].to_i512();
            let imm = (args[3].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_mask_srlv_epi64(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srai_epi32",
        vec![ArgType::I512, ArgType::U16, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srai_epi32 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u16();
            let a = args[2].to_i512();
            let imm = (args[3].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_mask_srav_epi32(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srai_epi64",
        vec![ArgType::I512, ArgType::U8, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srai_epi64 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u8();
            let a = args[2].to_i512();
            let imm = (args[3].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_mask_srav_epi64(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_slli_epi32",
        vec![ArgType::U16, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_slli_epi32 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u16();
            let a = args[1].to_i512();
            let imm = (args[2].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_maskz_sllv_epi32(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_slli_epi64",
        vec![ArgType::U8, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_slli_epi64 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u8();
            let a = args[1].to_i512();
            let imm = (args[2].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_maskz_sllv_epi64(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srli_epi32",
        vec![ArgType::U16, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srli_epi32 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u16();
            let a = args[1].to_i512();
            let imm = (args[2].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_maskz_srlv_epi32(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srli_epi64",
        vec![ArgType::U8, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srli_epi64 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u8();
            let a = args[1].to_i512();
            let imm = (args[2].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_maskz_srlv_epi64(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srai_epi32",
        vec![ArgType::U16, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srai_epi32 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u16();
            let a = args[1].to_i512();
            let imm = (args[2].to_u32() & 0x1f) as i32;
            let counts = unsafe { _mm512_set1_epi32(imm) };
            let res = unsafe { _mm512_maskz_srav_epi32(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srai_epi64",
        vec![ArgType::U8, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srai_epi64 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u8();
            let a = args[1].to_i512();
            let imm = (args[2].to_u32() & 0x3f) as i64;
            let counts = unsafe { _mm512_set1_epi64(imm) };
            let res = unsafe { _mm512_maskz_srav_epi64(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    // variable shift operations
    registry.register_instruction(Instruction::new(
        "_mm512_sllv_epi32",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_sllv_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let counts = args[1].to_i512();
            let res = unsafe { _mm512_sllv_epi32(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_sllv_epi64",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_sllv_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let counts = args[1].to_i512();
            let res = unsafe { _mm512_sllv_epi64(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srlv_epi32",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_srlv_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let counts = args[1].to_i512();
            let res = unsafe { _mm512_srlv_epi32(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srlv_epi64",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_srlv_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let counts = args[1].to_i512();
            let res = unsafe { _mm512_srlv_epi64(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srav_epi32",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 2 {
                return Err("_mm512_srav_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let counts = args[1].to_i512();
            let res = unsafe { _mm512_srav_epi32(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_srav_epi64",
        vec![ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 2 {
                return Err("_mm512_srav_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i512();
            let counts = args[1].to_i512();
            let res = unsafe { _mm512_srav_epi64(a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    // variable shift mask operations
    registry.register_instruction(Instruction::new(
        "_mm512_mask_sllv_epi32",
        vec![ArgType::I512, ArgType::U16, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 4 {
                return Err("_mm512_mask_sllv_epi32 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u16();
            let a = args[2].to_i512();
            let counts = args[3].to_i512();
            let res = unsafe { _mm512_mask_sllv_epi32(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_sllv_epi64",
        vec![ArgType::I512, ArgType::U8, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 4 {
                return Err("_mm512_mask_sllv_epi64 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u8();
            let a = args[2].to_i512();
            let counts = args[3].to_i512();
            let res = unsafe { _mm512_mask_sllv_epi64(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srlv_epi32",
        vec![ArgType::I512, ArgType::U16, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srlv_epi32 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u16();
            let a = args[2].to_i512();
            let counts = args[3].to_i512();
            let res = unsafe { _mm512_mask_srlv_epi32(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srlv_epi64",
        vec![ArgType::I512, ArgType::U8, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srlv_epi64 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u8();
            let a = args[2].to_i512();
            let counts = args[3].to_i512();
            let res = unsafe { _mm512_mask_srlv_epi64(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srav_epi32",
        vec![ArgType::I512, ArgType::U16, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srav_epi32 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u16();
            let a = args[2].to_i512();
            let counts = args[3].to_i512();
            let res = unsafe { _mm512_mask_srav_epi32(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_mask_srav_epi64",
        vec![ArgType::I512, ArgType::U8, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 4 {
                return Err("_mm512_mask_srav_epi64 requires 4 arguments".to_string());
            }
            let src = args[0].to_i512();
            let mask = args[1].to_u8();
            let a = args[2].to_i512();
            let counts = args[3].to_i512();
            let res = unsafe { _mm512_mask_srav_epi64(src, mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    // variable shift maskz operations
    registry.register_instruction(Instruction::new(
        "_mm512_maskz_sllv_epi32",
        vec![ArgType::U16, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_sllv_epi32 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u16();
            let a = args[1].to_i512();
            let counts = args[2].to_i512();
            let res = unsafe { _mm512_maskz_sllv_epi32(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_sllv_epi64",
        vec![ArgType::U8, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_sllv_epi64 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u8();
            let a = args[1].to_i512();
            let counts = args[2].to_i512();
            let res = unsafe { _mm512_maskz_sllv_epi64(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srlv_epi32",
        vec![ArgType::U16, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srlv_epi32 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u16();
            let a = args[1].to_i512();
            let counts = args[2].to_i512();
            let res = unsafe { _mm512_maskz_srlv_epi32(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srlv_epi64",
        vec![ArgType::U8, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srlv_epi64 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u8();
            let a = args[1].to_i512();
            let counts = args[2].to_i512();
            let res = unsafe { _mm512_maskz_srlv_epi64(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srav_epi32",
        vec![ArgType::U16, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srav_epi32 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u16();
            let a = args[1].to_i512();
            let counts = args[2].to_i512();
            let res = unsafe { _mm512_maskz_srav_epi32(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_maskz_srav_epi64",
        vec![ArgType::U8, ArgType::I512, ArgType::I512],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512dq()?;
            if args.len() != 3 {
                return Err("_mm512_maskz_srav_epi64 requires 3 arguments".to_string());
            }
            let mask = args[0].to_u8();
            let a = args[1].to_i512();
            let counts = args[2].to_i512();
            let res = unsafe { _mm512_maskz_srav_epi64(mask, a, counts) };
            Ok(m512i_to_argument(res))
        },
    ));
}
