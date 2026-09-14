use std::ffi::OsStr;
use std::io;

/// Apply service startup settings before the provider or native worker starts.
/// The C core and Rust must observe the same configured values on Windows.
pub(crate) fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) -> io::Result<()> {
    let key = key.as_ref();
    let value = value.as_ref();
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;

        unsafe extern "C" {
            fn qrt_server_set_environment_utf16_v1(name: *const u16, value: *const u16) -> i32;
        }

        let mut name: Vec<u16> = key.encode_wide().collect();
        let mut text: Vec<u16> = value.encode_wide().collect();
        if name.is_empty()
            || name.contains(&0)
            || name.contains(&(b'=' as u16))
            || text.contains(&0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid runtime environment key or value",
            ));
        }
        name.push(0);
        text.push(0);
        let status = unsafe { qrt_server_set_environment_utf16_v1(name.as_ptr(), text.as_ptr()) };
        if status != 0 {
            return Err(io::Error::other(format!(
                "native runtime environment assignment failed with status {status}"
            )));
        }
    }
    // Keep Rust's process-environment semantics, including a present empty
    // value. The CRT represents that value as an absent entry, matching the
    // native CLI; both disable value-based runtime flags. This also preserves
    // the exact OS string after the CRT's narrow/wide conversion.
    std::env::set_var(key, value);
    Ok(())
}
