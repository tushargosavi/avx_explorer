pub trait ToBitString {
    fn to_bit_string(&self) -> String;
}

pub fn to_bit_string<T: ToBitString + ?Sized>(data: &T) -> String {
    data.to_bit_string()
}

impl ToBitString for [u8] {
    fn to_bit_string(&self) -> String {
        self.iter()
            .map(|byte| format!("{:08b}", byte))
            .collect::<Vec<String>>()
            .join("_")
    }
}

macro_rules! impl_to_bit_string_for_int {
    ($($t:ty),*) => {
        $(
            impl ToBitString for $t {
                fn to_bit_string(&self) -> String {
                    self.to_be_bytes().to_bit_string()
                }
            }
        )*
    };
}

impl_to_bit_string_for_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl ToBitString for String {
    fn to_bit_string(&self) -> String {
        self.as_bytes().to_bit_string()
    }
}

impl ToBitString for str {
    fn to_bit_string(&self) -> String {
        self.as_bytes().to_bit_string()
    }
}

impl<const N: usize> ToBitString for [u8; N] {
    fn to_bit_string(&self) -> String {
        self.as_slice().to_bit_string()
    }
}

impl<T: ToBitString> ToBitString for &[T] {
    fn to_bit_string(&self) -> String {
        self.iter().map(|t| t.to_bit_string()).collect::<String>()
    }
}

pub enum HexChunkWidth {
    U8,
    U16,
    U32,
    U64,
}

pub fn to_hex_string(value: u64, chunk_width: HexChunkWidth) -> String {
    match chunk_width {
        HexChunkWidth::U8 => {
            let mut s = String::new();
            for i in 0..8 {
                s.push_str(&format!("0x{:02x} ", (value >> (56 - i * 8)) as u8));
            }
            s.trim_end().to_string()
        }
        HexChunkWidth::U16 => {
            let mut s = String::new();
            for i in 0..4 {
                s.push_str(&format!("0x{:04x} ", (value >> (48 - i * 16)) as u16));
            }
            s.trim_end().to_string()
        }
        HexChunkWidth::U32 => {
            let mut s = String::new();
            for i in 0..2 {
                s.push_str(&format!("0x{:08x} ", (value >> (32 - i * 32)) as u32));
            }
            s.trim_end().to_string()
        }
        HexChunkWidth::U64 => {
            format!("0x{:016x}", value)
        }
    }
}

pub fn to_decimal_string(value: u64, chunk_width: HexChunkWidth) -> String {
    match chunk_width {
        HexChunkWidth::U8 => {
            let mut s = String::new();
            for i in 0..8 {
                s.push_str(&format!("{} ", (value >> (56 - i * 8)) as u8));
            }
            s.trim_end().to_string()
        }
        HexChunkWidth::U16 => {
            let mut s = String::new();
            for i in 0..4 {
                s.push_str(&format!("{} ", (value >> (48 - i * 16)) as u16));
            }
            s.trim_end().to_string()
        }
        HexChunkWidth::U32 => {
            let mut s = String::new();
            for i in 0..2 {
                s.push_str(&format!("{} ", (value >> (32 - i * 32)) as u32));
            }
            s.trim_end().to_string()
        }
        HexChunkWidth::U64 => {
            format!("{}", value)
        }
    }
}
