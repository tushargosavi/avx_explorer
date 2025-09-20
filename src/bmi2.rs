use crate::ast::{ArgType, Argument, FunctionRegistry, Instruction};
use std::arch::x86_64::*;

#[inline]
fn require_bmi2() -> Result<(), String> {
    if !is_x86_feature_detected!("bmi2") {
        Err("BMI2 not supported on this CPU/runtime".to_string())
    } else {
        Ok(())
    }
}

pub fn register_bmi2_instructions(registry: &mut FunctionRegistry) {
    // _pdep_u32(src, mask) -> u32
    registry.register_instruction(Instruction::new(
        "_pdep_u32",
        vec![ArgType::U32, ArgType::U32],
        ArgType::U32,
        |_, args| {
            require_bmi2()?;
            if args.len() != 2 {
                return Err("_pdep_u32 requires exactly 2 arguments".to_string());
            }
            let src = args[0].to_u32();
            let mask = args[1].to_u32();
            let res = unsafe { _pdep_u32(src, mask) } as u64;
            Ok(Argument::Scalar(res))
        },
    ));

    // _pdep_u64(src, mask) -> u64
    registry.register_instruction(Instruction::new(
        "_pdep_u64",
        vec![ArgType::U64, ArgType::U64],
        ArgType::U64,
        |_, args| {
            require_bmi2()?;
            if args.len() != 2 {
                return Err("_pdep_u64 requires exactly 2 arguments".to_string());
            }
            let src = args[0].to_u64();
            let mask = args[1].to_u64();
            let res = unsafe { _pdep_u64(src, mask) };
            Ok(Argument::Scalar(res))
        },
    ));
}


