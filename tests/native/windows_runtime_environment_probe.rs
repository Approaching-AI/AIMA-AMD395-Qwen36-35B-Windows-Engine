#[path = "../../engine/qrt-server/src/environment.rs"]
mod environment;

use std::ffi::CStr;
use std::os::raw::c_char;

unsafe extern "C" {
    fn probe_native_early_prefill_enabled() -> i32;
    fn probe_get_crt_early_prefill() -> *const c_char;
    fn probe_crt_wide_matches(name: *const u16, expected: *const u16) -> i32;
    fn probe_invalid_native_arguments() -> i32;
}

const KEY: &str = "QRT_QWEN36_WHOLE_PROVIDER_EARLY_PREFILL_STREAM_CALLBACK";

fn observe(stage: &str, expected_os: &str, expected_crt: Option<&str>, enabled: bool) {
    let os = std::env::var(KEY).unwrap();
    let pointer = unsafe { probe_get_crt_early_prefill() };
    let crt = if pointer.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(pointer) }
                .to_str()
                .unwrap()
                .to_owned(),
        )
    };
    let actual_enabled = unsafe { probe_native_early_prefill_enabled() } != 0;
    assert_eq!(os, expected_os);
    assert_eq!(crt.as_deref(), expected_crt);
    assert_eq!(actual_enabled, enabled);
    println!("{{\"kind\":\"windows_runtime_environment_fixed\",\"stage\":{stage:?},\"os_value\":{os:?},\"crt_value\":{},\"actual_core_early_prefill_enabled\":{actual_enabled},\"pass\":true}}",
        crt.as_ref().map(|v| format!("{v:?}")).unwrap_or("null".to_owned()));
}

fn wide(name: &str, value: Option<&str>) -> bool {
    let name: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
    let value: Option<Vec<u16>> = value.map(|s| s.encode_utf16().chain(Some(0)).collect());
    unsafe {
        probe_crt_wide_matches(
            name.as_ptr(),
            value.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
        ) != 0
    }
}

fn main() {
    assert!(cfg!(windows));
    assert!(std::env::var(KEY).is_err());
    std::env::set_var(KEY, "1");
    observe(
        "original_rust_assignment_reproduces_stale_core",
        "1",
        None,
        false,
    );
    environment::set_var(KEY, "1").unwrap();
    observe(
        "production_setter_enables_actual_core",
        "1",
        Some("1"),
        true,
    );
    environment::set_var(KEY, "0").unwrap();
    observe(
        "production_override_disables_actual_core",
        "0",
        Some("0"),
        false,
    );
    environment::set_var(KEY, "").unwrap();
    observe("empty_process_value_clears_crt_flag", "", None, false);
    environment::set_var(KEY, "1").unwrap();
    observe(
        "repeated_assignment_enables_actual_core",
        "1",
        Some("1"),
        true,
    );
    for (key, value) in [
        ("", "1"),
        ("BAD=KEY", "1"),
        ("BAD\0KEY", "1"),
        (KEY, "1\0bad"),
    ] {
        assert!(environment::set_var(key, value).is_err());
    }
    assert_eq!(unsafe { probe_invalid_native_arguments() }, 1);
    observe(
        "invalid_inputs_leave_current_flag_unchanged",
        "1",
        Some("1"),
        true,
    );
    let unicode_key = "QRT_TEST_RUNTIME_ENVIRONMENT_UNICODE";
    let unicode_value = "D:\\模型 路径\\😀\\provider.dll";
    environment::set_var(unicode_key, unicode_value).unwrap();
    assert_eq!(std::env::var(unicode_key).unwrap(), unicode_value);
    assert!(wide(unicode_key, Some(unicode_value)));
    let long_value = "x".repeat(8192);
    environment::set_var(unicode_key, &long_value).unwrap();
    assert_eq!(std::env::var(unicode_key).unwrap(), long_value);
    assert!(wide(unicode_key, Some(&long_value)));
    environment::set_var(unicode_key, "").unwrap();
    assert_eq!(std::env::var(unicode_key).unwrap(), "");
    assert!(wide(unicode_key, None));
    environment::set_var(KEY, "").unwrap();
    observe("final_empty_value_disables_actual_core", "", None, false);
    println!("{{\"kind\":\"windows_runtime_environment_fixed_summary\",\"core_cases\":7,\"invalid_rust_inputs\":4,\"invalid_native_inputs\":4,\"unicode_wide_value_match\":true,\"long_value_units\":8192,\"empty_value_semantics_preserved\":true,\"production_rust_setter_used\":true,\"actual_c_core_reader_used\":true,\"model_loaded\":false,\"gpu_work_submitted\":false}}");
}
