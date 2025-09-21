#[cfg(test)]
mod tests {
    use crate::ast::{AST, ArgType, Argument};
    use crate::interpreter::Interpreter;
    use crate::parser::parse_input;

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
