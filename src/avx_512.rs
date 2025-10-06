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

#[inline]
fn require_avx512bw() -> Result<(), String> {
    if !is_x86_feature_detected!("avx512bw") {
        Err("AVX-512BW not supported on this CPU/runtime".to_string())
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

    // align operations
    registry.register_instruction(Instruction::new(
        "_mm512_alignr_epi8",
        vec![ArgType::I512, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512bw()?;
            if args.len() != 3 {
                return Err("_mm512_alignr_epi8 requires 3 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let imm = (args[2].to_u32() & 0xFF) as i32;
            let res = unsafe {
                match imm {
                    0 => _mm512_alignr_epi8::<0>(a, b),
                    1 => _mm512_alignr_epi8::<1>(a, b),
                    2 => _mm512_alignr_epi8::<2>(a, b),
                    3 => _mm512_alignr_epi8::<3>(a, b),
                    4 => _mm512_alignr_epi8::<4>(a, b),
                    5 => _mm512_alignr_epi8::<5>(a, b),
                    6 => _mm512_alignr_epi8::<6>(a, b),
                    7 => _mm512_alignr_epi8::<7>(a, b),
                    8 => _mm512_alignr_epi8::<8>(a, b),
                    9 => _mm512_alignr_epi8::<9>(a, b),
                    10 => _mm512_alignr_epi8::<10>(a, b),
                    11 => _mm512_alignr_epi8::<11>(a, b),
                    12 => _mm512_alignr_epi8::<12>(a, b),
                    13 => _mm512_alignr_epi8::<13>(a, b),
                    14 => _mm512_alignr_epi8::<14>(a, b),
                    15 => _mm512_alignr_epi8::<15>(a, b),
                    16 => _mm512_alignr_epi8::<16>(a, b),
                    17 => _mm512_alignr_epi8::<17>(a, b),
                    18 => _mm512_alignr_epi8::<18>(a, b),
                    19 => _mm512_alignr_epi8::<19>(a, b),
                    20 => _mm512_alignr_epi8::<20>(a, b),
                    21 => _mm512_alignr_epi8::<21>(a, b),
                    22 => _mm512_alignr_epi8::<22>(a, b),
                    23 => _mm512_alignr_epi8::<23>(a, b),
                    24 => _mm512_alignr_epi8::<24>(a, b),
                    25 => _mm512_alignr_epi8::<25>(a, b),
                    26 => _mm512_alignr_epi8::<26>(a, b),
                    27 => _mm512_alignr_epi8::<27>(a, b),
                    28 => _mm512_alignr_epi8::<28>(a, b),
                    29 => _mm512_alignr_epi8::<29>(a, b),
                    30 => _mm512_alignr_epi8::<30>(a, b),
                    31 => _mm512_alignr_epi8::<31>(a, b),
                    32 => _mm512_alignr_epi8::<32>(a, b),
                    33 => _mm512_alignr_epi8::<33>(a, b),
                    34 => _mm512_alignr_epi8::<34>(a, b),
                    35 => _mm512_alignr_epi8::<35>(a, b),
                    36 => _mm512_alignr_epi8::<36>(a, b),
                    37 => _mm512_alignr_epi8::<37>(a, b),
                    38 => _mm512_alignr_epi8::<38>(a, b),
                    39 => _mm512_alignr_epi8::<39>(a, b),
                    40 => _mm512_alignr_epi8::<40>(a, b),
                    41 => _mm512_alignr_epi8::<41>(a, b),
                    42 => _mm512_alignr_epi8::<42>(a, b),
                    43 => _mm512_alignr_epi8::<43>(a, b),
                    44 => _mm512_alignr_epi8::<44>(a, b),
                    45 => _mm512_alignr_epi8::<45>(a, b),
                    46 => _mm512_alignr_epi8::<46>(a, b),
                    47 => _mm512_alignr_epi8::<47>(a, b),
                    48 => _mm512_alignr_epi8::<48>(a, b),
                    49 => _mm512_alignr_epi8::<49>(a, b),
                    50 => _mm512_alignr_epi8::<50>(a, b),
                    51 => _mm512_alignr_epi8::<51>(a, b),
                    52 => _mm512_alignr_epi8::<52>(a, b),
                    53 => _mm512_alignr_epi8::<53>(a, b),
                    54 => _mm512_alignr_epi8::<54>(a, b),
                    55 => _mm512_alignr_epi8::<55>(a, b),
                    56 => _mm512_alignr_epi8::<56>(a, b),
                    57 => _mm512_alignr_epi8::<57>(a, b),
                    58 => _mm512_alignr_epi8::<58>(a, b),
                    59 => _mm512_alignr_epi8::<59>(a, b),
                    60 => _mm512_alignr_epi8::<60>(a, b),
                    61 => _mm512_alignr_epi8::<61>(a, b),
                    62 => _mm512_alignr_epi8::<62>(a, b),
                    63 => _mm512_alignr_epi8::<63>(a, b),
                    64 => _mm512_alignr_epi8::<64>(a, b),
                    _ => _mm512_setzero_si512(),
                }
            };
            Ok(m512i_to_argument(res))
        },
    ));

    registry.register_instruction(Instruction::new(
        "_mm512_alignl_epi8",
        vec![ArgType::I512, ArgType::I512, ArgType::U32],
        ArgType::I512,
        |_, args| {
            require_avx512f()?;
            require_avx512bw()?;
            if args.len() != 3 {
                return Err("_mm512_alignl_epi8 requires 3 arguments".to_string());
            }
            let a = args[0].to_i512();
            let b = args[1].to_i512();
            let imm = (args[2].to_u32() & 0xFF) as i32;
            let res = unsafe {
                match imm {
                    0 => _mm512_alignr_epi8::<0>(b, a),
                    1 => _mm512_alignr_epi8::<1>(b, a),
                    2 => _mm512_alignr_epi8::<2>(b, a),
                    3 => _mm512_alignr_epi8::<3>(b, a),
                    4 => _mm512_alignr_epi8::<4>(b, a),
                    5 => _mm512_alignr_epi8::<5>(b, a),
                    6 => _mm512_alignr_epi8::<6>(b, a),
                    7 => _mm512_alignr_epi8::<7>(b, a),
                    8 => _mm512_alignr_epi8::<8>(b, a),
                    9 => _mm512_alignr_epi8::<9>(b, a),
                    10 => _mm512_alignr_epi8::<10>(b, a),
                    11 => _mm512_alignr_epi8::<11>(b, a),
                    12 => _mm512_alignr_epi8::<12>(b, a),
                    13 => _mm512_alignr_epi8::<13>(b, a),
                    14 => _mm512_alignr_epi8::<14>(b, a),
                    15 => _mm512_alignr_epi8::<15>(b, a),
                    16 => _mm512_alignr_epi8::<16>(b, a),
                    17 => _mm512_alignr_epi8::<17>(b, a),
                    18 => _mm512_alignr_epi8::<18>(b, a),
                    19 => _mm512_alignr_epi8::<19>(b, a),
                    20 => _mm512_alignr_epi8::<20>(b, a),
                    21 => _mm512_alignr_epi8::<21>(b, a),
                    22 => _mm512_alignr_epi8::<22>(b, a),
                    23 => _mm512_alignr_epi8::<23>(b, a),
                    24 => _mm512_alignr_epi8::<24>(b, a),
                    25 => _mm512_alignr_epi8::<25>(b, a),
                    26 => _mm512_alignr_epi8::<26>(b, a),
                    27 => _mm512_alignr_epi8::<27>(b, a),
                    28 => _mm512_alignr_epi8::<28>(b, a),
                    29 => _mm512_alignr_epi8::<29>(b, a),
                    30 => _mm512_alignr_epi8::<30>(b, a),
                    31 => _mm512_alignr_epi8::<31>(b, a),
                    32 => _mm512_alignr_epi8::<32>(b, a),
                    33 => _mm512_alignr_epi8::<33>(b, a),
                    34 => _mm512_alignr_epi8::<34>(b, a),
                    35 => _mm512_alignr_epi8::<35>(b, a),
                    36 => _mm512_alignr_epi8::<36>(b, a),
                    37 => _mm512_alignr_epi8::<37>(b, a),
                    38 => _mm512_alignr_epi8::<38>(b, a),
                    39 => _mm512_alignr_epi8::<39>(b, a),
                    40 => _mm512_alignr_epi8::<40>(b, a),
                    41 => _mm512_alignr_epi8::<41>(b, a),
                    42 => _mm512_alignr_epi8::<42>(b, a),
                    43 => _mm512_alignr_epi8::<43>(b, a),
                    44 => _mm512_alignr_epi8::<44>(b, a),
                    45 => _mm512_alignr_epi8::<45>(b, a),
                    46 => _mm512_alignr_epi8::<46>(b, a),
                    47 => _mm512_alignr_epi8::<47>(b, a),
                    48 => _mm512_alignr_epi8::<48>(b, a),
                    49 => _mm512_alignr_epi8::<49>(b, a),
                    50 => _mm512_alignr_epi8::<50>(b, a),
                    51 => _mm512_alignr_epi8::<51>(b, a),
                    52 => _mm512_alignr_epi8::<52>(b, a),
                    53 => _mm512_alignr_epi8::<53>(b, a),
                    54 => _mm512_alignr_epi8::<54>(b, a),
                    55 => _mm512_alignr_epi8::<55>(b, a),
                    56 => _mm512_alignr_epi8::<56>(b, a),
                    57 => _mm512_alignr_epi8::<57>(b, a),
                    58 => _mm512_alignr_epi8::<58>(b, a),
                    59 => _mm512_alignr_epi8::<59>(b, a),
                    60 => _mm512_alignr_epi8::<60>(b, a),
                    61 => _mm512_alignr_epi8::<61>(b, a),
                    62 => _mm512_alignr_epi8::<62>(b, a),
                    63 => _mm512_alignr_epi8::<63>(b, a),
                    64 => _mm512_alignr_epi8::<64>(b, a),
                    _ => _mm512_setzero_si512(),
                }
            };
            Ok(m512i_to_argument(res))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mm512_alignr_by_imm(a: __m512i, b: __m512i, imm: i32) -> __m512i {
        unsafe {
            match imm {
                0 => _mm512_alignr_epi8::<0>(a, b),
                1 => _mm512_alignr_epi8::<1>(a, b),
                2 => _mm512_alignr_epi8::<2>(a, b),
                3 => _mm512_alignr_epi8::<3>(a, b),
                4 => _mm512_alignr_epi8::<4>(a, b),
                5 => _mm512_alignr_epi8::<5>(a, b),
                6 => _mm512_alignr_epi8::<6>(a, b),
                7 => _mm512_alignr_epi8::<7>(a, b),
                8 => _mm512_alignr_epi8::<8>(a, b),
                9 => _mm512_alignr_epi8::<9>(a, b),
                10 => _mm512_alignr_epi8::<10>(a, b),
                11 => _mm512_alignr_epi8::<11>(a, b),
                12 => _mm512_alignr_epi8::<12>(a, b),
                13 => _mm512_alignr_epi8::<13>(a, b),
                14 => _mm512_alignr_epi8::<14>(a, b),
                15 => _mm512_alignr_epi8::<15>(a, b),
                16 => _mm512_alignr_epi8::<16>(a, b),
                17 => _mm512_alignr_epi8::<17>(a, b),
                18 => _mm512_alignr_epi8::<18>(a, b),
                19 => _mm512_alignr_epi8::<19>(a, b),
                20 => _mm512_alignr_epi8::<20>(a, b),
                21 => _mm512_alignr_epi8::<21>(a, b),
                22 => _mm512_alignr_epi8::<22>(a, b),
                23 => _mm512_alignr_epi8::<23>(a, b),
                24 => _mm512_alignr_epi8::<24>(a, b),
                25 => _mm512_alignr_epi8::<25>(a, b),
                26 => _mm512_alignr_epi8::<26>(a, b),
                27 => _mm512_alignr_epi8::<27>(a, b),
                28 => _mm512_alignr_epi8::<28>(a, b),
                29 => _mm512_alignr_epi8::<29>(a, b),
                30 => _mm512_alignr_epi8::<30>(a, b),
                31 => _mm512_alignr_epi8::<31>(a, b),
                32 => _mm512_alignr_epi8::<32>(a, b),
                33 => _mm512_alignr_epi8::<33>(a, b),
                34 => _mm512_alignr_epi8::<34>(a, b),
                35 => _mm512_alignr_epi8::<35>(a, b),
                36 => _mm512_alignr_epi8::<36>(a, b),
                37 => _mm512_alignr_epi8::<37>(a, b),
                38 => _mm512_alignr_epi8::<38>(a, b),
                39 => _mm512_alignr_epi8::<39>(a, b),
                40 => _mm512_alignr_epi8::<40>(a, b),
                41 => _mm512_alignr_epi8::<41>(a, b),
                42 => _mm512_alignr_epi8::<42>(a, b),
                43 => _mm512_alignr_epi8::<43>(a, b),
                44 => _mm512_alignr_epi8::<44>(a, b),
                45 => _mm512_alignr_epi8::<45>(a, b),
                46 => _mm512_alignr_epi8::<46>(a, b),
                47 => _mm512_alignr_epi8::<47>(a, b),
                48 => _mm512_alignr_epi8::<48>(a, b),
                49 => _mm512_alignr_epi8::<49>(a, b),
                50 => _mm512_alignr_epi8::<50>(a, b),
                51 => _mm512_alignr_epi8::<51>(a, b),
                52 => _mm512_alignr_epi8::<52>(a, b),
                53 => _mm512_alignr_epi8::<53>(a, b),
                54 => _mm512_alignr_epi8::<54>(a, b),
                55 => _mm512_alignr_epi8::<55>(a, b),
                56 => _mm512_alignr_epi8::<56>(a, b),
                57 => _mm512_alignr_epi8::<57>(a, b),
                58 => _mm512_alignr_epi8::<58>(a, b),
                59 => _mm512_alignr_epi8::<59>(a, b),
                60 => _mm512_alignr_epi8::<60>(a, b),
                61 => _mm512_alignr_epi8::<61>(a, b),
                62 => _mm512_alignr_epi8::<62>(a, b),
                63 => _mm512_alignr_epi8::<63>(a, b),
                64 => _mm512_alignr_epi8::<64>(a, b),
                _ => _mm512_setzero_si512(),
            }
        }
    }

    #[test]
    fn alignr_i512_produces_expected_bytes() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let mut a_bytes = [0u8; 64];
        let mut b_bytes = [0u8; 64];
        for i in 0..64 {
            a_bytes[i] = 0x10 + i as u8;
            b_bytes[i] = 0x80 + i as u8;
        }

        let a = unsafe { std::mem::transmute::<[u8; 64], __m512i>(a_bytes) };
        let b = unsafe { std::mem::transmute::<[u8; 64], __m512i>(b_bytes) };

        let res = mm512_alignr_by_imm(a, b, 12);
        let bytes = m512i_to_bytes(res);
        let expected: [u8; 64] = [
            140, 141, 142, 143, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 156, 157, 158, 159,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 172, 173, 174, 175, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 188, 189, 190, 191, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
            74, 75,
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn alignl_i512_produces_expected_bytes() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let mut a_bytes = [0u8; 64];
        let mut b_bytes = [0u8; 64];
        for i in 0..64 {
            a_bytes[i] = 0x10 + i as u8;
            b_bytes[i] = 0x80 + i as u8;
        }

        let a = unsafe { std::mem::transmute::<[u8; 64], __m512i>(a_bytes) };
        let b = unsafe { std::mem::transmute::<[u8; 64], __m512i>(b_bytes) };

        let res = mm512_alignr_by_imm(b, a, 20);
        let bytes = m512i_to_bytes(res);
        let expected: [u8; 64] = [
            132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 0, 0, 0, 0, 148, 149, 150,
            151, 152, 153, 154, 155, 156, 157, 158, 159, 0, 0, 0, 0, 164, 165, 166, 167, 168, 169,
            170, 171, 172, 173, 174, 175, 0, 0, 0, 0, 180, 181, 182, 183, 184, 185, 186, 187, 188,
            189, 190, 191, 0, 0, 0, 0,
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn align_i512_zeroes_when_shift_exceeds_length() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let a = unsafe { std::mem::transmute::<[u8; 64], __m512i>([1u8; 64]) };
        let b = unsafe { std::mem::transmute::<[u8; 64], __m512i>([2u8; 64]) };

        let res = mm512_alignr_by_imm(a, b, 200);
        let bytes = m512i_to_bytes(res);
        assert!(bytes.iter().all(|&byte| byte == 0));
    }
}
