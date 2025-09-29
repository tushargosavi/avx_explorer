#[cfg(test)]
mod tests {
    use crate::ast::{AST, ArgType, Argument};
    use crate::interpreter::{Interpreter, MemoryEntry, RegisterEntry, argument_to_utf8_lossy};
    use crate::parser::parse_input;
    use std::convert::TryInto;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_parse_simple_function_call() {
        let result = parse_input("test()");
        assert!(result.is_ok());

        match result.unwrap() {
            AST::Call { name, args } => {
                assert_eq!(name, "test");
                assert!(args.is_empty());
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_parse_function_with_scalar_args() {
        let result = parse_input("add(0x10, 20)");
        assert!(result.is_ok());

        match result.unwrap() {
            AST::Call { name, args } => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Argument::Scalar(16));
                assert_eq!(args[1], Argument::Scalar(20));
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_parse_array_arguments() {
        let result = parse_input("add(w[0x123, 0x456], qw[1, 2, 3, 4])");
        assert!(result.is_ok());

        match result.unwrap() {
            AST::Call { name, args } => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);

                match &args[0] {
                    Argument::Array(ArgType::I256, values) => {
                        let expected_bytes = [0x23u8, 0x01u8, 0x56u8, 0x04u8];
                        assert_eq!(&values[..4], &expected_bytes);
                    }
                    _ => panic!("Expected I256 array"),
                }

                match &args[1] {
                    Argument::Array(ArgType::I256, values) => {
                        let expected_bytes = [
                            1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                            0u8, 0u8, 3u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8,
                            0u8, 0u8, 0u8, 0u8,
                        ];
                        assert_eq!(&values[..32], &expected_bytes);
                    }
                    _ => panic!("Expected I256 array"),
                }
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_parse_assignment() {
        let result = parse_input("result = add(5, 10)");
        assert!(result.is_ok());

        match result.unwrap() {
            AST::Assign { dest, child } => {
                assert_eq!(dest, "result");
                match *child {
                    AST::Call { name, args } => {
                        assert_eq!(name, "add");
                        assert_eq!(args.len(), 2);
                        assert_eq!(args[0], Argument::Scalar(5));
                        assert_eq!(args[1], Argument::Scalar(10));
                    }
                    _ => panic!("Expected Call in assignment"),
                }
            }
            _ => panic!("Expected Assign"),
        }
    }

    #[test]
    fn test_parse_variable_declaration() {
        let result = parse_input("x = 42");
        assert!(result.is_ok());

        match result.unwrap() {
            AST::Var { name, value } => {
                assert_eq!(name, "x");
                assert_eq!(value, Argument::Scalar(42));
            }
            _ => panic!("Expected Var"),
        }
    }

    #[test]
    fn test_parse_different_number_formats() {
        let test_cases = vec![
            ("test(0xFF)", 255),
            ("test(0o77)", 63),
            ("test(0b1010)", 10),
            ("test(123)", 123),
        ];

        for (input, expected) in test_cases {
            let result = parse_input(input);
            assert!(result.is_ok(), "Failed to parse: {}", input);

            match result.unwrap() {
                AST::Call { args, .. } => {
                    assert_eq!(args.len(), 1);
                    assert_eq!(args[0], Argument::Scalar(expected));
                }
                _ => panic!("Expected Call"),
            }
        }
    }

    #[test]
    fn test_parse_string_literal_argument() {
        let result = parse_input("message = \"Hi\\n\"").unwrap();
        match result {
            AST::Var { name, value } => {
                assert_eq!(name, "message");
                assert_eq!(value, Argument::Memory(b"Hi\n".to_vec()));
            }
            other => panic!("Expected variable assignment, got {:?}", other),
        }
    }

    #[test]
    fn test_ast_validate_unknown_function() {
        let interpreter = Interpreter::new();
        let ast = AST::Call {
            name: "does_not_exist".to_string(),
            args: vec![],
        };

        let result = ast.validate(&interpreter.function_registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_ast_validate_argument_counts() {
        let interpreter = Interpreter::new();

        let too_few = AST::Call {
            name: "add".to_string(),
            args: vec![Argument::Scalar(1)],
        };
        assert!(too_few.validate(&interpreter.function_registry).is_err());

        let correct = AST::Call {
            name: "add".to_string(),
            args: vec![Argument::Scalar(1), Argument::Scalar(2)],
        };
        assert!(correct.validate(&interpreter.function_registry).is_ok());

        let too_many = AST::Call {
            name: "print_hex".to_string(),
            args: vec![
                Argument::Scalar(1),
                Argument::Scalar(32),
                Argument::Scalar(64),
            ],
        };
        assert!(too_many.validate(&interpreter.function_registry).is_err());
    }

    #[test]
    fn test_argument_conversions() {
        let scalar = Argument::Scalar(0x123456789ABCDEF0);
        let array = Argument::Array(ArgType::I512, {
            let mut bytes = [0u8; 64];
            bytes[0] = 1;
            bytes[8] = 2;
            bytes[16] = 3;
            bytes[24] = 4;
            bytes[32] = 5;
            bytes[40] = 6;
            bytes[48] = 7;
            bytes[56] = 8;
            bytes
        });

        assert_eq!(scalar.to_u64(), 0x123456789ABCDEF0);
        assert_eq!(scalar.to_u32(), 0x9ABCDEF0);
        assert_eq!(scalar.to_u16(), 0xDEF0);

        let i256_result = scalar.to_i256();
        let i256_array: [i32; 8] = unsafe { std::mem::transmute(i256_result) };
        assert_eq!(i256_array[0] as u32, 0x9ABCDEF0u32);

        let i512_result = scalar.to_i512();
        let i512_array: [i64; 8] = unsafe { std::mem::transmute(i512_result) };
        assert_eq!(i512_array[0], 0x123456789ABCDEF0i64);

        assert_eq!(array.to_u64(), 1);
        assert_eq!(array.to_u32(), 1);
        assert_eq!(array.to_u16(), 1);
    }

    #[test]
    fn test_mm512_loadu_si512_returns_register() {
        let mut interpreter = Interpreter::new();
        let input: Vec<u8> = (0u8..64u8).collect();
        interpreter
            .variables
            .insert("input".to_string(), Argument::Memory(input.clone()));

        let ast = AST::Assign {
            dest: "data".to_string(),
            child: Box::new(AST::Call {
                name: "_mm512_loadu_si512".to_string(),
                args: vec![Argument::Variable("input".to_string())],
            }),
        };

        let result = interpreter.execute(ast).expect("load should succeed");

        let expected_bytes = input;
        match result {
            Argument::Array(ArgType::I512, bytes) => {
                assert_eq!(&bytes[..], expected_bytes.as_slice());
            }
            other => panic!("Expected vector register, got {:?}", other),
        }

        match interpreter.variables.get("data") {
            Some(Argument::Array(ArgType::I512, bytes)) => {
                assert_eq!(&bytes[..], expected_bytes.as_slice());
            }
            other => panic!("Expected stored register, got {:?}", other),
        }

        match interpreter.variables.get("_res") {
            Some(Argument::Array(ArgType::I512, bytes)) => {
                assert_eq!(&bytes[..], expected_bytes.as_slice());
            }
            other => panic!(
                "Expected _res to contain the loaded register, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_mm512_loadu_si512_zero_pads_short_memory() {
        let mut interpreter = Interpreter::new();
        let input: Vec<u8> = (0u8..50u8).collect();
        interpreter
            .variables
            .insert("input".to_string(), Argument::Memory(input.clone()));

        let ast = AST::Assign {
            dest: "data".to_string(),
            child: Box::new(AST::Call {
                name: "_mm512_loadu_si512".to_string(),
                args: vec![Argument::Variable("input".to_string())],
            }),
        };

        let result = interpreter.execute(ast).expect("load should succeed");

        let mut expected_bytes = [0u8; 64];
        expected_bytes[..input.len()].copy_from_slice(&input);

        let assert_loaded = |arg: &Argument| match arg {
            Argument::Array(ArgType::I512, bytes) => {
                assert_eq!(bytes.len(), 64);
                assert_eq!(&bytes[..], &expected_bytes);
            }
            other => panic!("Expected vector register, got {:?}", other),
        };

        assert_loaded(&result);
        assert_loaded(
            interpreter
                .variables
                .get("data")
                .expect("variable should be stored"),
        );
        assert_loaded(
            interpreter
                .variables
                .get("_res")
                .expect("_res should contain result"),
        );
    }

    #[test]
    fn test_mm256_loadu_si256_zero_pads_short_memory() {
        let mut interpreter = Interpreter::new();
        let input: Vec<u8> = (0u8..20u8).collect();
        interpreter
            .variables
            .insert("input".to_string(), Argument::Memory(input.clone()));

        let ast = AST::Assign {
            dest: "data".to_string(),
            child: Box::new(AST::Call {
                name: "_mm256_loadu_si256".to_string(),
                args: vec![Argument::Variable("input".to_string())],
            }),
        };

        let result = interpreter.execute(ast).expect("load should succeed");

        let mut expected_bytes = [0u8; 64];
        expected_bytes[..input.len()].copy_from_slice(&input);
        let expected_len = ArgType::I256.vector_byte_len();

        let assert_loaded = |arg: &Argument| match arg {
            Argument::Array(ArgType::I256, bytes) => {
                assert_eq!(&bytes[..expected_len], &expected_bytes[..expected_len]);
            }
            other => panic!("Expected vector register, got {:?}", other),
        };

        assert_loaded(&result);
        assert_loaded(
            interpreter
                .variables
                .get("data")
                .expect("variable should be stored"),
        );
        assert_loaded(
            interpreter
                .variables
                .get("_res")
                .expect("_res should contain result"),
        );
    }

    #[test]
    fn test_mem_initializer_with_type() {
        let mut interpreter = Interpreter::new();
        let ast = parse_input("a = mem[0x1u32, 0x23, 0x23, 0x4]").unwrap();
        let result = interpreter.execute(ast).unwrap();
        match result {
            Argument::Memory(bytes) => {
                assert_eq!(bytes.len(), 16);
                assert_eq!(&bytes[0..4], &[0x01, 0x00, 0x00, 0x00]);
                assert_eq!(&bytes[4..8], &[0x23, 0x00, 0x00, 0x00]);
            }
            other => panic!("Expected memory, got {:?}", other),
        }
    }

    #[test]
    fn test_zero_initializer() {
        let mut interpreter = Interpreter::new();
        let ast = parse_input("buf = zero[10]").unwrap();
        let result = interpreter.execute(ast).unwrap();
        match result {
            Argument::Memory(bytes) => {
                assert_eq!(bytes.len(), 10);
                assert!(bytes.iter().all(|&b| b == 0));
            }
            other => panic!("Expected memory, got {:?}", other),
        }
    }

    #[test]
    fn test_range_initializer_default_steps() {
        let mut interpreter = Interpreter::new();

        let ast = parse_input("buf = range(0, 4)").unwrap();
        let result = interpreter.execute(ast).unwrap();
        match result {
            Argument::Memory(bytes) => {
                assert_eq!(bytes, vec![0, 1, 2, 3, 4]);
            }
            other => panic!("Expected memory, got {:?}", other),
        }

        let ast = parse_input("down = range(5, 1)").unwrap();
        let result = interpreter.execute(ast).unwrap();
        match result {
            Argument::Memory(bytes) => {
                assert_eq!(bytes, vec![5, 4, 3, 2, 1]);
            }
            other => panic!("Expected memory, got {:?}", other),
        }
    }

    #[test]
    fn test_range_initializer_with_type_and_step() {
        let mut interpreter = Interpreter::new();
        let ast = parse_input("words = range(0u16, 6, 2)").unwrap();
        let result = interpreter.execute(ast).unwrap();
        match result {
            Argument::Memory(bytes) => {
                assert_eq!(bytes.len(), 8);
                assert_eq!(bytes, vec![0, 0, 2, 0, 4, 0, 6, 0]);
            }
            other => panic!("Expected memory, got {:?}", other),
        }
    }

    #[test]
    fn test_memory_slice_assignment_and_lookup() {
        let mut interpreter = Interpreter::new();
        interpreter
            .execute(parse_input("buf = mem[16]").unwrap())
            .unwrap();
        interpreter
            .execute(parse_input("buf[0] = 0x1122u16").unwrap())
            .unwrap();
        interpreter
            .execute(parse_input("buf[2..6] = [0x33, 0x44, 0x55, 0x66]").unwrap())
            .unwrap();
        let slice = interpreter
            .execute(parse_input("buf[0..6]").unwrap())
            .unwrap();
        match slice {
            Argument::Memory(bytes) => {
                assert_eq!(bytes, vec![0x22, 0x11, 0x33, 0x44, 0x55, 0x66]);
            }
            other => panic!("Expected memory slice, got {:?}", other),
        }
    }

    #[test]
    fn test_sse2_add_epi32() {
        let mut interpreter = Interpreter::new();

        let set_vec = AST::Call {
            name: "_mm_set1_epi32".to_string(),
            args: vec![Argument::Scalar(5)],
        };
        let vec_value = interpreter
            .execute(set_vec)
            .expect("_mm_set1_epi32 should execute");

        match &vec_value {
            Argument::Array(arg_type, bytes) => {
                assert_eq!(*arg_type, ArgType::I128);
                let slice: [u8; 16] = bytes[..16].try_into().unwrap();
                let data: [i32; 4] = unsafe { std::mem::transmute(slice) };
                assert!(data.iter().all(|v| *v == 5));
            }
            _ => panic!("Expected I128 array from _mm_set1_epi32"),
        }

        let add_ast = AST::Call {
            name: "_mm_add_epi32".to_string(),
            args: vec![vec_value.clone(), vec_value.clone()],
        };
        let result = interpreter
            .execute(add_ast)
            .expect("_mm_add_epi32 should execute");

        match result {
            Argument::Array(arg_type, bytes) => {
                assert_eq!(arg_type, ArgType::I128);
                let slice: [u8; 16] = bytes[..16].try_into().unwrap();
                let data: [i32; 4] = unsafe { std::mem::transmute(slice) };
                assert!(data.iter().all(|v| *v == 10));
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_argument_debug_shows_valid_bytes() {
        let mut raw = [0u8; 64];
        for (i, byte) in raw.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let expected_slice: Vec<u8> = raw[..32].to_vec();
        let arg = Argument::Array(ArgType::I256, raw);
        let debug = format!("{:?}", arg);
        let expected = format!("Array({:?}, {:?})", ArgType::I256, &expected_slice[..]);
        assert_eq!(debug, expected);
    }

    #[test]
    fn test_argument_to_utf8_lossy_helper() {
        let mut raw = [0u8; 64];
        let message = b"Hello";
        raw[..message.len()].copy_from_slice(message);
        let arg = Argument::Array(ArgType::I256, raw);
        let text = argument_to_utf8_lossy(&arg).unwrap();
        assert_eq!(text, "Hello");

        let mem_arg = Argument::Memory(b"Hi there".to_vec());
        let mem_text = argument_to_utf8_lossy(&mem_arg).unwrap();
        assert_eq!(mem_text, "Hi there");

        let scalar_val = u16::from_le_bytes(*b"Ok") as u64;
        let scalar_arg = Argument::ScalarTyped(ArgType::U16, scalar_val);
        let scalar_text = argument_to_utf8_lossy(&scalar_arg).unwrap();
        assert_eq!(scalar_text, "Ok");
    }

    #[test]
    fn test_file_function_reads_file_into_memory() {
        let mut interpreter = Interpreter::new();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let file_path = std::env::temp_dir().join(format!(
            "avx_explorer_test_{}_{}",
            std::process::id(),
            timestamp
        ));
        let data = b"File contents for testing";
        fs::write(&file_path, data).expect("unable to write test file");

        let input = format!("buf = file(\"{}\")", file_path.display());
        let ast = parse_input(&input).expect("file() call should parse");
        let result = interpreter
            .execute(ast)
            .expect("file() call should execute");

        match result {
            Argument::Memory(bytes) => assert_eq!(bytes, data),
            other => panic!("Expected memory result from file(), got {:?}", other),
        }

        fs::remove_file(&file_path).expect("unable to clean up test file");
    }

    #[test]
    fn test_interpreter_variable_storage() {
        let mut interpreter = Interpreter::new();

        let ast = AST::Var {
            name: "x".to_string(),
            value: Argument::Scalar(42),
        };

        let result = interpreter.execute(ast).unwrap();
        assert_eq!(result, Argument::Scalar(42));

        let call_ast = AST::Call {
            name: "test".to_string(),
            args: vec![Argument::Variable("x".to_string())],
        };

        let result = interpreter.execute(call_ast).unwrap();
        assert_eq!(result, Argument::Scalar(42));
    }

    #[test]
    fn test_interpreter_assignment() {
        let mut interpreter = Interpreter::new();

        let ast = AST::Assign {
            dest: "result".to_string(),
            child: Box::new(AST::Call {
                name: "add".to_string(),
                args: vec![Argument::Scalar(10), Argument::Scalar(20)],
            }),
        };

        let result = interpreter.execute(ast).unwrap();
        assert_eq!(result, Argument::Scalar(30));

        let x = interpreter.variables.get("result").unwrap();
        assert_eq!(x, &Argument::Scalar(30));
    }

    #[test]
    fn test_interpreter_add_function() {
        let mut interpreter = Interpreter::new();

        let ast = AST::Call {
            name: "add".to_string(),
            args: vec![Argument::Scalar(15), Argument::Scalar(27)],
        };

        let result = interpreter.execute(ast).unwrap();
        assert_eq!(result, Argument::Scalar(42));
    }

    #[test]
    fn test_environment_snapshot_splits_memory_and_registers() {
        let mut interpreter = Interpreter::new();
        interpreter
            .variables
            .insert("buf".to_string(), Argument::Memory(vec![1, 2, 3, 4, 5]));
        interpreter
            .variables
            .insert("count".to_string(), Argument::Scalar(7));

        let snapshot = interpreter.environment_snapshot();

        assert_eq!(
            snapshot.memory,
            vec![MemoryEntry {
                name: "buf".to_string(),
                length: 5,
            }]
        );

        assert_eq!(
            snapshot.registers,
            vec![RegisterEntry {
                name: "count".to_string(),
                type_name: "Scalar".to_string(),
                value: "7".to_string(),
            }]
        );
    }

    #[test]
    fn test_environment_snapshot_array_preview_truncates() {
        let mut interpreter = Interpreter::new();
        let mut array_bytes = [0u8; 64];
        for (idx, byte) in array_bytes.iter_mut().enumerate() {
            *byte = idx as u8;
        }
        interpreter
            .variables
            .insert("arr".to_string(), Argument::Array(ArgType::U8, array_bytes));

        let snapshot = interpreter.environment_snapshot();
        let entry = snapshot
            .registers
            .iter()
            .find(|entry| entry.name == "arr")
            .expect("array entry should be present");

        assert_eq!(entry.type_name, "Array<U8>");
        assert!(entry.value.starts_with("b[0, 1, 2, 3"));
        assert!(entry.value.ends_with("…]"));
    }

    #[test]
    fn test_complex_example() {
        let input = "_mm256_shuffle_epi8(w[0x00,0x12,0x13,0x43],w[0xFF,0x01,0xFF,0x83])";
        let result = parse_input(input);

        assert!(result.is_ok());

        match result.unwrap() {
            AST::Call { name, args } => {
                assert_eq!(name, "_mm256_shuffle_epi8");
                assert_eq!(args.len(), 2);

                match &args[0] {
                    Argument::Array(ArgType::I256, values) => {
                        let expected_bytes = [
                            0x00u8, 0x00u8, 0x12u8, 0x00u8, 0x13u8, 0x00u8, 0x43u8, 0x00u8,
                        ];
                        assert_eq!(&values[..8], &expected_bytes);
                    }
                    _ => panic!("Expected I256 array"),
                }

                match &args[1] {
                    Argument::Array(ArgType::I256, values) => {
                        let expected_bytes = [
                            0xFFu8, 0x00u8, 0x01u8, 0x00u8, 0xFFu8, 0x00u8, 0x83u8, 0x00u8,
                        ];
                        assert_eq!(&values[..8], &expected_bytes);
                    }
                    _ => panic!("Expected I256 array"),
                }
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_bit_parsing() {
        let input = "test(bits[1,0,1])";
        let result = parse_input(input);

        assert!(result.is_ok());

        match result.unwrap() {
            AST::Call { name, args } => {
                assert_eq!(name, "test");
                assert_eq!(args.len(), 1);

                match &args[0] {
                    Argument::Array(ArgType::I512, values) => {
                        // bits[1,0,1] should become 0b101 = 5 in little endian
                        assert_eq!((*values)[0], 0b101);
                    }
                    _ => panic!("Expected I512 array"),
                }
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_bit_parsing_long() {
        let input = "test(bits[1,0,1,1,0,0,1,0,1,0])";
        let result = parse_input(input);

        assert!(result.is_ok());

        match result.unwrap() {
            AST::Call { name, args } => {
                assert_eq!(name, "test");
                assert_eq!(args.len(), 1);

                match &args[0] {
                    Argument::Array(ArgType::I512, values) => {
                        // bits[1,0,1,1,0,0,1,0] should become 0b01001101 = 0x4D in little endian
                        // bits[1,0] should become 0b01 = 1 in little endian
                        assert_eq!((*values)[0], 0b01001101);
                        assert_eq!((*values)[1], 0b01);
                    }
                    _ => panic!("Expected I512 array"),
                }
            }
            _ => panic!("Expected Call"),
        }
    }
}
