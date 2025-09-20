use crate::ast::{ArgType, Argument, FunctionRegistry, Instruction};
use std::arch::x86_64::*;

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
}
