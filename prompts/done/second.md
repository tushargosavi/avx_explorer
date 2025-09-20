After implementing the parser, now its time to implement the interpreter with required functionality.
add method to convert Argument to avx register values.

First implement/correct implementation of following functions, the 512 and 256 types are from
std::arch::x86_64 module. while converting if the array or source values do not fit the destination
then fill the values with zeros.

```
impl Argument {
	fn to_i256(&self) -> __m256i;
	fn to_i512(&self) -> __m512i;
	fn to_u64(&self) -> u64;
	fn to_u32(&self) -> u32;
	fn to_u16(&self) -> u16;
}
```

the types are declared in https://doc.rust-lang.org/core/arch/x86_64/index.html

Define function registery with following information
```
enum ArgType {
	I256,
	I512,
	U64,
	U32,
	U16,
	U8,
	Ptr
}

struct FunctionRegistry {
    functions: Vec<FunctionInfo>,
}

struct FunctionInfo {
	name: String, // name of the function
	arguments: Vec<ArgType>, // type of argument accepted
	return_type: ArgType
}
```

for now register following functions
```
- _mm256_mask_expand_epi8
- _mm256_mask_expand_epi16
- _mm256_mask_expand_epi32
- _mm256_mask_expand_epi64
```

the result values are converted back to the Argument type for assigment statement and stored in global mutable
interpreter state, and printed on the repl.
