use crate::ast::*;
use std::arch::x86_64::*;
use std::collections::HashMap;

pub struct Interpreter {
    pub variables: HashMap<String, Argument>,
    pub function_registry: FunctionRegistry,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut registry = FunctionRegistry::new();

        // Register the AVX functions
        registry.register(
            "_mm256_mask_expand_epi8",
            vec![ArgType::I256, ArgType::I256, ArgType::U8],
            ArgType::I256,
        );
        registry.register(
            "_mm256_mask_expand_epi16",
            vec![ArgType::I256, ArgType::I256, ArgType::U16],
            ArgType::I256,
        );
        registry.register(
            "_mm256_mask_expand_epi32",
            vec![ArgType::I256, ArgType::I256, ArgType::U32],
            ArgType::I256,
        );
        registry.register(
            "_mm256_mask_expand_epi64",
            vec![ArgType::I256, ArgType::I256, ArgType::U64],
            ArgType::I256,
        );

        Self {
            variables: HashMap::new(),
            function_registry: registry,
        }
    }

    pub fn execute(&mut self, ast: AST) -> Result<Argument, String> {
        match ast {
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
        }
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

        println!("Resolved arguments: {:?}", resolved_args);
        match name.as_str() {
            "add" => self.execute_add(&resolved_args),
            "test" => Ok(Argument::Scalar(42)),
            "_mm256_mask_expand_epi8" => self.execute_mm256_mask_expand_epi8(&resolved_args),
            "_mm256_mask_expand_epi16" => self.execute_mm256_mask_expand_epi16(&resolved_args),
            "_mm256_mask_expand_epi32" => self.execute_mm256_mask_expand_epi32(&resolved_args),
            "_mm256_mask_expand_epi64" => self.execute_mm256_mask_expand_epi64(&resolved_args),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    fn execute_add(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 2 {
            return Err("add requires exactly 2 arguments".to_string());
        }

        let a = args[0].to_u64();
        let b = args[1].to_u64();

        Ok(Argument::Scalar(a + b))
    }

    fn execute_mm256_mask_expand_epi8(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi8 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u8() as u32;

        let result = unsafe { _mm256_mask_expand_epi8(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi16(&self, args: &[Argument]) -> Result<Argument, String> {
        println!("Executing _mm256_mask_expand_epi16 {:?}", args);
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi16 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u16() as u16;

        let result = unsafe { _mm256_mask_expand_epi16(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi32(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi32 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u32() as u8;

        let result = unsafe { _mm256_mask_expand_epi32(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn execute_mm256_mask_expand_epi64(&self, args: &[Argument]) -> Result<Argument, String> {
        if args.len() != 3 {
            return Err("_mm256_mask_expand_epi64 requires exactly 3 arguments".to_string());
        }

        let src = args[0].to_i256();
        let mask = args[1].to_i256();
        let k = args[2].to_u64() as u8;

        let result = unsafe { _mm256_mask_expand_epi64(src, k, mask) };
        Ok(self.m256i_to_argument(result))
    }

    fn m256i_to_argument(&self, value: __m256i) -> Argument {
        let array: [i32; 8] = unsafe { std::mem::transmute(value) };
        Argument::Array(AType::DoubleWord, array.iter().map(|&x| x as u64).collect())
    }
}
