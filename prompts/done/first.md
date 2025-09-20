create a rust project to execute avx2 and avx512 instructions as repl for experimenting and 
understanding their efforts.

```
enum AType {
	Bit, Byte, Word, DoubleWord, QuadWord
}


enum Argument {
	Array(atype: AType, values: Vec<u64>) // type in case of array
	Memory(Vec<u8>) // for pointer type arguments to the function.
	Scalar(u64), // type in case of scalar
}

impl Argument {
	fn to_i256(&self) -> _m_256i;
	fn to_i512(&self) -> _m_512i;
	fn to_u64(&self) -> u64;
	fn to_u32(&self) -> u32;
	fn to_u16(&self) -> u16;
}
```

The sample syntax looks like below

```
call = ident '(' args ')'
stmt = call
	| ident '=' call
args = /* empty */
	| array 
	| num
	| ident
array = qualifier '[' values ']'
qualifier = 'bits' | 'b' | 'w' | 'dw' | 'qw'
values = /* empty */
	| num values
num = integer in hex  (0x123) or octal 'O023' or binary b0100101
```

ident is identifier token and num is Numeric value token


A sample input on prompt looks like below
```
prompt> _mm256_shuffle_epi8(w[0x00,0x12,0x13,0x43],w[0xFF,0x01,0xFF,0x83])
```

The array prefix are
- B for bits (bits)
- b for byte
- w for word  (16 bit int)
- dw for double word (32 bit int)
- qw for 64 bit ints

qw[0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8] is a 512 bit value, as each element is 
64 bit values, and total number of elements are 8.


```
enum AST {
	Call(
		name: String,
		args: Vec<Argument>
	),
	Var(
		name: String,
		value: Argument
	),
	Assign(
		dest: String,
		child: Call
	)
}
```


create a parser using chumsky to parse this small language, and later we will extend to add
more function to emulate the functionality.

