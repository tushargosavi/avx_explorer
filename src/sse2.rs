use crate::ast::{ArgType, Argument, FunctionRegistry, Instruction};
use std::arch::x86_64::*;

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

#[inline]
fn require_sse2() -> Result<(), String> {
    if !is_x86_feature_detected!("sse2") {
        Err("SSE2 not supported on this CPU/runtime".to_string())
    } else {
        Ok(())
    }
}

pub fn register_sse2_instructions(registry: &mut FunctionRegistry) {
    // set1/broadcast
    registry.register_instruction(Instruction::new(
        "_mm_set1_epi8",
        vec![ArgType::U8],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 1 {
                return Err("_mm_set1_epi8 requires 1 argument".to_string());
            }
            let v = args[0].to_u8() as i8;
            let res = unsafe { _mm_set1_epi8(v) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_set1_epi16",
        vec![ArgType::U16],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 1 {
                return Err("_mm_set1_epi16 requires 1 argument".to_string());
            }
            let v = args[0].to_u16() as i16;
            let res = unsafe { _mm_set1_epi16(v) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_set1_epi32",
        vec![ArgType::U32],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 1 {
                return Err("_mm_set1_epi32 requires 1 argument".to_string());
            }
            let v = args[0].to_u32() as i32;
            let res = unsafe { _mm_set1_epi32(v) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_set1_epi64x",
        vec![ArgType::U64],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
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
        "_mm_add_epi8",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_add_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_add_epi8(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_add_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_add_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_add_epi16(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_add_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_add_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_add_epi32(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_add_epi64",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_add_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_add_epi64(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));

    // sub
    registry.register_instruction(Instruction::new(
        "_mm_sub_epi8",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sub_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_sub_epi8(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sub_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sub_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_sub_epi16(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sub_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sub_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_sub_epi32(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sub_epi64",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sub_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_sub_epi64(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));

    // multiplication
    registry.register_instruction(Instruction::new(
        "_mm_mullo_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_mullo_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_mullo_epi16(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_mul_epu32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_mul_epu32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_mul_epu32(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));

    // bitwise
    registry.register_instruction(Instruction::new(
        "_mm_and_si128",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_and_si128 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_and_si128(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_or_si128",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_or_si128 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_or_si128(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_xor_si128",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_xor_si128 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_xor_si128(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_andnot_si128",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_andnot_si128 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_andnot_si128(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));

    // compares
    registry.register_instruction(Instruction::new(
        "_mm_cmpeq_epi8",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpeq_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpeq_epi8(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cmpeq_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpeq_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpeq_epi16(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cmpeq_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpeq_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpeq_epi32(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cmpeq_epi64",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpeq_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpeq_epi64(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cmpgt_epi8",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpgt_epi8 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpgt_epi8(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cmpgt_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpgt_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpgt_epi16(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_cmpgt_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_cmpgt_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let b = args[1].to_i128();
            let res = unsafe { _mm_cmpgt_epi32(a, b) };
            Ok(m128i_to_argument(res))
        },
    ));

    // shifts (vector-count variants)
    registry.register_instruction(Instruction::new(
        "_mm_sll_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sll_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_sll_epi16(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sll_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sll_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_sll_epi32(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sll_epi64",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sll_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_sll_epi64(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_srl_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_srl_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_srl_epi16(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_srl_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_srl_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_srl_epi32(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_srl_epi64",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_srl_epi64 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_srl_epi64(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sra_epi16",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sra_epi16 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_sra_epi16(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
    registry.register_instruction(Instruction::new(
        "_mm_sra_epi32",
        vec![ArgType::I128, ArgType::I128],
        ArgType::I128,
        |_, args| {
            require_sse2()?;
            if args.len() != 2 {
                return Err("_mm_sra_epi32 requires 2 arguments".to_string());
            }
            let a = args[0].to_i128();
            let count = args[1].to_i128();
            let res = unsafe { _mm_sra_epi32(a, count) };
            Ok(m128i_to_argument(res))
        },
    ));
}
