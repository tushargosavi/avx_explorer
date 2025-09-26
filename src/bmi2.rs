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
            Ok(Argument::ScalarTyped(ArgType::U32, res))
        },
    ));

    // _pext_u32(src, mask) -> u32
    registry.register_instruction(Instruction::new(
        "_pext_u32",
        vec![ArgType::U32, ArgType::U32],
        ArgType::U32,
        |_, args| {
            require_bmi2()?;
            if args.len() != 2 {
                return Err("_pext_u32 requires exactly 2 arguments".to_string());
            }
            let src = args[0].to_u32();
            let mask = args[1].to_u32();
            let res = unsafe { _pext_u32(src, mask) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U32, res))
        },
    ));

    // _bzhi_u32(src, index) -> u32
    registry.register_instruction(Instruction::new(
        "_bzhi_u32",
        vec![ArgType::U32, ArgType::U32],
        ArgType::U32,
        |_, args| {
            require_bmi2()?;
            if args.len() != 2 {
                return Err("_bzhi_u32 requires exactly 2 arguments".to_string());
            }
            let src = args[0].to_u32();
            let index = args[1].to_u32();
            let res = unsafe { _bzhi_u32(src, index) } as u64;
            Ok(Argument::ScalarTyped(ArgType::U32, res))
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
            Ok(Argument::ScalarTyped(ArgType::U64, res))
        },
    ));

    // _pext_u64(src, mask) -> u64
    registry.register_instruction(Instruction::new(
        "_pext_u64",
        vec![ArgType::U64, ArgType::U64],
        ArgType::U64,
        |_, args| {
            require_bmi2()?;
            if args.len() != 2 {
                return Err("_pext_u64 requires exactly 2 arguments".to_string());
            }
            let src = args[0].to_u64();
            let mask = args[1].to_u64();
            let res = unsafe { _pext_u64(src, mask) };
            Ok(Argument::ScalarTyped(ArgType::U64, res))
        },
    ));

    // _bzhi_u64(src, index) -> u64
    registry.register_instruction(Instruction::new(
        "_bzhi_u64",
        vec![ArgType::U64, ArgType::U64],
        ArgType::U64,
        |_, args| {
            require_bmi2()?;
            if args.len() != 2 {
                return Err("_bzhi_u64 requires exactly 2 arguments".to_string());
            }
            let src = args[0].to_u64();
            let index = args[1].to_u32();
            let res = unsafe { _bzhi_u64(src, index) };
            Ok(Argument::ScalarTyped(ArgType::U64, res))
        },
    ));

    // _mulx_u32(a, b, &mut hi) -> u32
    registry.register_instruction(Instruction::new(
        "_mulx_u32",
        vec![ArgType::U32, ArgType::U32, ArgType::Ptr],
        ArgType::U32,
        |ctx, args| {
            require_bmi2()?;
            if args.len() != 3 {
                return Err("_mulx_u32 requires exactly 3 arguments".to_string());
            }
            let a = args[0].to_u32();
            let b = args[1].to_u32();
            let hi_var_name = match &args[2] {
                Argument::Variable(name) => name.clone(),
                other => {
                    return Err(format!(
                        "_mulx_u32 third argument must be a pointer to a variable, found {:?}",
                        other
                    ));
                }
            };

            if let Some(existing) = ctx.get_var(&hi_var_name) {
                match existing {
                    Argument::Scalar(_) | Argument::ScalarTyped(_, _) => {}
                    other => {
                        return Err(format!(
                            "Pointer '{}' must reference a scalar variable, found {:?}",
                            hi_var_name, other
                        ));
                    }
                }
            }

            let mut hi_out: u32 = 0;
            let lo = unsafe { _mulx_u32(a, b, &mut hi_out) } as u64;
            ctx.set_var(
                &hi_var_name,
                Argument::ScalarTyped(ArgType::U32, hi_out as u64),
            );
            Ok(Argument::ScalarTyped(ArgType::U32, lo))
        },
    ));

    // _mulx_u64(a, b, &mut hi) -> u64
    registry.register_instruction(Instruction::new(
        "_mulx_u64",
        vec![ArgType::U64, ArgType::U64, ArgType::Ptr],
        ArgType::U64,
        |ctx, args| {
            require_bmi2()?;
            if args.len() != 3 {
                return Err("_mulx_u64 requires exactly 3 arguments".to_string());
            }
            let a = args[0].to_u64();
            let b = args[1].to_u64();
            let hi_var_name = match &args[2] {
                Argument::Variable(name) => name.clone(),
                other => {
                    return Err(format!(
                        "_mulx_u64 third argument must be a pointer to a variable, found {:?}",
                        other
                    ));
                }
            };

            if let Some(existing) = ctx.get_var(&hi_var_name) {
                match existing {
                    Argument::Scalar(_) | Argument::ScalarTyped(_, _) => {}
                    other => {
                        return Err(format!(
                            "Pointer '{}' must reference a scalar variable, found {:?}",
                            hi_var_name, other
                        ));
                    }
                }
            }

            let mut hi_out: u64 = 0;
            let lo = unsafe { _mulx_u64(a, b, &mut hi_out) };
            ctx.set_var(&hi_var_name, Argument::ScalarTyped(ArgType::U64, hi_out));
            Ok(Argument::ScalarTyped(ArgType::U64, lo))
        },
    ));
}
