use crate::ast::{ArgType, Argument, FunctionRegistry, Instruction};
use std::arch::x86_64::*;

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
}
