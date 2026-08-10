use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr::NonNull;
use std::sync::Mutex;

use thiserror::Error;
use tokio::sync::mpsc;

pub const NATIVE_MAX_OUTPUT_TOKENS: usize = 512;
pub const NATIVE_THREAD_STACK_BYTES: usize = 256 * 1024 * 1024;
const QRT_LOAD_ERROR_CAPACITY: usize = 192;
const QRT_FAILURE_STAGE_CAPACITY: usize = 64;
const QRT_SERVER_BRIDGE_ABI_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct BackendToken {
    pub index: usize,
    pub token_id: u32,
    pub phase: u32,
    pub token_step_ns: u64,
    pub request_elapsed_ns: u64,
}

#[derive(Clone, Debug, Default)]
pub struct LoadMetrics {
    pub provider_dll_load_ns: u64,
    pub provider_preload_wall_ns: u64,
    pub provider_preload_reported_ns: u64,
    pub provider_preload_stored_entries: u64,
    pub engine_create_ns: u64,
    pub total_load_ns: u64,
}

#[derive(Clone, Debug, Default)]
pub struct RequestMetrics {
    pub queue_wait_ns: u64,
    pub wall_ns: u64,
    pub ttft_ns: u64,
    pub tpot_ns: u64,
    pub tpot_samples: usize,
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub token_ids: Vec<u32>,
    pub metrics: RequestMetrics,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("{field} contains an embedded NUL byte")]
    InvalidPath { field: &'static str },
    #[error("native engine load failed ({status}) at {stage}: {message}")]
    Load {
        status: &'static str,
        stage: String,
        message: String,
    },
    #[error("native request failed ({status}) at {stage}: {message}")]
    Request {
        status: &'static str,
        stage: String,
        message: String,
    },
    #[error("native engine serialization lock was poisoned")]
    LockPoisoned,
    #[error("native engine returned an invalid output length: {0}")]
    InvalidOutputLength(usize),
    #[error("could not start the native execution thread: {0}")]
    ThreadStart(String),
    #[error("the native execution thread panicked")]
    ThreadPanic,
}

pub trait InferenceBackend: Send + Sync {
    fn generate(
        &self,
        input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
    ) -> Result<GenerationResult, BackendError>;

    fn generate_exact_first_token(
        &self,
        input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
    ) -> Result<GenerationResult, BackendError> {
        self.generate(input_token_ids, max_output_tokens, events)
    }

    fn max_output_tokens(&self) -> usize;
}

#[repr(C)]
struct QrtServerEngine {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QrtServerLoadReport {
    struct_size: u32,
    abi_version: u32,
    provider_dll_load_ns: u64,
    provider_preload_wall_ns: u64,
    provider_preload_reported_ns: u64,
    provider_preload_stored_entry_count: u64,
    engine_create_ns: u64,
    total_load_ns: u64,
    provider_preload_completed: u32,
    engine_ready: u32,
    failure_stage: [c_char; QRT_FAILURE_STAGE_CAPACITY],
    failure: [c_char; QRT_LOAD_ERROR_CAPACITY],
}

impl Default for QrtServerLoadReport {
    fn default() -> Self {
        Self {
            struct_size: 0,
            abi_version: 0,
            provider_dll_load_ns: 0,
            provider_preload_wall_ns: 0,
            provider_preload_reported_ns: 0,
            provider_preload_stored_entry_count: 0,
            engine_create_ns: 0,
            total_load_ns: 0,
            provider_preload_completed: 0,
            engine_ready: 0,
            failure_stage: [0; QRT_FAILURE_STAGE_CAPACITY],
            failure: [0; QRT_LOAD_ERROR_CAPACITY],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QrtServerRequestReport {
    struct_size: u32,
    abi_version: u32,
    request_wall_ns: u64,
    ttft_ns: u64,
    tpot_ns: u64,
    tpot_sample_count: usize,
    output_token_count: usize,
    failure_stage: [c_char; QRT_FAILURE_STAGE_CAPACITY],
    failure: [c_char; QRT_LOAD_ERROR_CAPACITY],
}

impl Default for QrtServerRequestReport {
    fn default() -> Self {
        Self {
            struct_size: 0,
            abi_version: 0,
            request_wall_ns: 0,
            ttft_ns: 0,
            tpot_ns: 0,
            tpot_sample_count: 0,
            output_token_count: 0,
            failure_stage: [0; QRT_FAILURE_STAGE_CAPACITY],
            failure: [0; QRT_LOAD_ERROR_CAPACITY],
        }
    }
}

#[repr(C)]
struct QrtTokenStreamEvent {
    struct_size: u32,
    abi_version: u32,
    phase: u32,
    output_index: u32,
    token_id: u32,
    reserved0: u32,
    token_step_elapsed_ns: u64,
    request_elapsed_ns: u64,
    provider_decode_elapsed_ns: u64,
}

type QrtTokenCallback = unsafe extern "C" fn(*mut c_void, *const QrtTokenStreamEvent) -> c_int;

unsafe extern "C" {
    fn qrt_server_engine_create_v1(
        model_path: *const c_char,
        provider_dll: *const c_char,
        max_context_tokens: usize,
        out_engine: *mut *mut QrtServerEngine,
        out_report: *mut QrtServerLoadReport,
    ) -> c_int;

    fn qrt_server_engine_free_v1(engine: *mut QrtServerEngine);

    fn qrt_server_engine_request_tokens_stream_v1(
        engine: *mut QrtServerEngine,
        input_tokens: *const u32,
        input_token_count: usize,
        output_tokens: *mut u32,
        output_token_capacity: usize,
        out_output_token_count: *mut usize,
        callback: QrtTokenCallback,
        user_data: *mut c_void,
        out_report: *mut QrtServerRequestReport,
    ) -> c_int;

    fn qrt_server_engine_request_tokens_exact_stream_v1(
        engine: *mut QrtServerEngine,
        input_tokens: *const u32,
        input_token_count: usize,
        output_tokens: *mut u32,
        output_token_capacity: usize,
        out_output_token_count: *mut usize,
        callback: QrtTokenCallback,
        user_data: *mut c_void,
        out_report: *mut QrtServerRequestReport,
    ) -> c_int;
}

struct CallbackContext {
    sender: Option<mpsc::UnboundedSender<BackendToken>>,
}

unsafe extern "C" fn stream_callback(
    user_data: *mut c_void,
    event: *const QrtTokenStreamEvent,
) -> c_int {
    if user_data.is_null() || event.is_null() {
        return 0;
    }
    let context = unsafe { &mut *(user_data.cast::<CallbackContext>()) };
    let event = unsafe { &*event };
    if event.abi_version != QRT_SERVER_BRIDGE_ABI_VERSION {
        return 0;
    }
    if let Some(sender) = &context.sender {
        return i32::from(
            sender
                .send(BackendToken {
                    index: event.output_index as usize,
                    token_id: event.token_id,
                    phase: event.phase,
                    token_step_ns: event.token_step_elapsed_ns,
                    request_elapsed_ns: event.request_elapsed_ns,
                })
                .is_ok(),
        );
    }
    1
}

pub struct NativeBackend {
    engine: NonNull<QrtServerEngine>,
    request_lock: Mutex<()>,
    load_metrics: LoadMetrics,
}

// The C engine owns no thread-local Rust state. Its public request entry is
// serialized both here and by the native critical section.
unsafe impl Send for NativeBackend {}
unsafe impl Sync for NativeBackend {}

impl NativeBackend {
    pub fn load(
        model_path: &str,
        provider_dll: &str,
        max_context_tokens: usize,
    ) -> Result<Self, BackendError> {
        let model_path = CString::new(model_path).map_err(|_| BackendError::InvalidPath {
            field: "model path",
        })?;
        let provider_dll = CString::new(provider_dll).map_err(|_| BackendError::InvalidPath {
            field: "provider DLL",
        })?;
        let mut raw_engine = std::ptr::null_mut();
        let mut report = QrtServerLoadReport::default();
        let status = unsafe {
            qrt_server_engine_create_v1(
                model_path.as_ptr(),
                provider_dll.as_ptr(),
                max_context_tokens,
                &mut raw_engine,
                &mut report,
            )
        };
        let Some(engine) = NonNull::new(raw_engine) else {
            return Err(BackendError::Load {
                status: status_name(status),
                stage: c_array_to_string(&report.failure_stage),
                message: c_array_to_string(&report.failure),
            });
        };
        if status != 0 || report.engine_ready == 0 {
            unsafe { qrt_server_engine_free_v1(engine.as_ptr()) };
            return Err(BackendError::Load {
                status: status_name(status),
                stage: c_array_to_string(&report.failure_stage),
                message: c_array_to_string(&report.failure),
            });
        }
        Ok(Self {
            engine,
            request_lock: Mutex::new(()),
            load_metrics: LoadMetrics {
                provider_dll_load_ns: report.provider_dll_load_ns,
                provider_preload_wall_ns: report.provider_preload_wall_ns,
                provider_preload_reported_ns: report.provider_preload_reported_ns,
                provider_preload_stored_entries: report.provider_preload_stored_entry_count,
                engine_create_ns: report.engine_create_ns,
                total_load_ns: report.total_load_ns,
            },
        })
    }

    pub fn load_metrics(&self) -> &LoadMetrics {
        &self.load_metrics
    }

    fn generate_with_route(
        &self,
        input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
        exact_first_token: bool,
    ) -> Result<GenerationResult, BackendError> {
        if input_token_ids.is_empty() || max_output_tokens == 0 {
            return Err(BackendError::Request {
                status: "invalid_argument",
                stage: "request_validation".to_owned(),
                message: "input and output token counts must be non-zero".to_owned(),
            });
        }
        if max_output_tokens > NATIVE_MAX_OUTPUT_TOKENS {
            return Err(BackendError::Request {
                status: "invalid_argument",
                stage: "request_validation".to_owned(),
                message: format!(
                    "max output tokens exceeds native capacity {NATIVE_MAX_OUTPUT_TOKENS}"
                ),
            });
        }

        std::thread::scope(|scope| {
            let worker = std::thread::Builder::new()
                .name("qrt-native-request".to_owned())
                .stack_size(NATIVE_THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    self.generate_on_native_thread(
                        input_token_ids,
                        max_output_tokens,
                        events,
                        exact_first_token,
                    )
                })
                .map_err(|error| BackendError::ThreadStart(error.to_string()))?;
            worker.join().map_err(|_| BackendError::ThreadPanic)?
        })
    }
}

impl InferenceBackend for NativeBackend {
    fn generate(
        &self,
        input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
    ) -> Result<GenerationResult, BackendError> {
        self.generate_with_route(input_token_ids, max_output_tokens, events, false)
    }

    fn generate_exact_first_token(
        &self,
        input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
    ) -> Result<GenerationResult, BackendError> {
        self.generate_with_route(input_token_ids, max_output_tokens, events, true)
    }

    fn max_output_tokens(&self) -> usize {
        NATIVE_MAX_OUTPUT_TOKENS
    }
}

impl NativeBackend {
    fn generate_on_native_thread(
        &self,
        input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
        exact_first_token: bool,
    ) -> Result<GenerationResult, BackendError> {
        let _guard = self
            .request_lock
            .lock()
            .map_err(|_| BackendError::LockPoisoned)?;
        let mut output = vec![0_u32; max_output_tokens];
        let mut output_count = 0_usize;
        let mut report = QrtServerRequestReport::default();
        let mut callback_context = CallbackContext { sender: events };
        let status = unsafe {
            let request = if exact_first_token {
                qrt_server_engine_request_tokens_exact_stream_v1
            } else {
                qrt_server_engine_request_tokens_stream_v1
            };
            request(
                self.engine.as_ptr(),
                input_token_ids.as_ptr(),
                input_token_ids.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut output_count,
                stream_callback,
                (&mut callback_context as *mut CallbackContext).cast(),
                &mut report,
            )
        };
        if status != 0 {
            return Err(BackendError::Request {
                status: status_name(status),
                stage: c_array_to_string(&report.failure_stage),
                message: c_array_to_string(&report.failure),
            });
        }
        if output_count > output.len() {
            return Err(BackendError::InvalidOutputLength(output_count));
        }
        output.truncate(output_count);
        Ok(GenerationResult {
            token_ids: output,
            metrics: RequestMetrics {
                queue_wait_ns: 0,
                wall_ns: report.request_wall_ns,
                ttft_ns: report.ttft_ns,
                tpot_ns: report.tpot_ns,
                tpot_samples: report.tpot_sample_count,
            },
        })
    }
}

impl Drop for NativeBackend {
    fn drop(&mut self) {
        unsafe { qrt_server_engine_free_v1(self.engine.as_ptr()) };
    }
}

fn status_name(status: c_int) -> &'static str {
    match status {
        0 => "ok",
        1 => "invalid_argument",
        2 => "out_of_memory",
        3 => "not_implemented",
        4 => "unsupported",
        5 => "io_error",
        6 => "parse_error",
        _ => "unknown",
    }
}

fn c_array_to_string<const N: usize>(value: &[c_char; N]) -> String {
    if value[0] == 0 {
        return String::new();
    }
    unsafe { CStr::from_ptr(value.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
pub struct ScriptedBackend {
    output: Vec<u32>,
    max_output: usize,
}

#[cfg(test)]
impl ScriptedBackend {
    pub fn new(output: Vec<u32>) -> Self {
        Self {
            output,
            max_output: NATIVE_MAX_OUTPUT_TOKENS,
        }
    }
}

#[cfg(test)]
impl InferenceBackend for ScriptedBackend {
    fn generate(
        &self,
        _input_token_ids: &[u32],
        max_output_tokens: usize,
        events: Option<mpsc::UnboundedSender<BackendToken>>,
    ) -> Result<GenerationResult, BackendError> {
        let output: Vec<u32> = self
            .output
            .iter()
            .copied()
            .take(max_output_tokens)
            .collect();
        if let Some(sender) = events {
            for (index, token_id) in output.iter().copied().enumerate() {
                let _ = sender.send(BackendToken {
                    index,
                    token_id,
                    phase: u32::from(index != 0),
                    token_step_ns: 1,
                    request_elapsed_ns: (index + 1) as u64,
                });
            }
        }
        Ok(GenerationResult {
            token_ids: output,
            metrics: RequestMetrics::default(),
        })
    }

    fn max_output_tokens(&self) -> usize {
        self.max_output
    }
}

#[cfg(test)]
mod callback_tests {
    use super::*;

    fn event() -> QrtTokenStreamEvent {
        QrtTokenStreamEvent {
            struct_size: std::mem::size_of::<QrtTokenStreamEvent>() as u32,
            abi_version: QRT_SERVER_BRIDGE_ABI_VERSION,
            phase: 0,
            output_index: 0,
            token_id: 42,
            reserved0: 0,
            token_step_elapsed_ns: 1,
            request_elapsed_ns: 1,
            provider_decode_elapsed_ns: 0,
        }
    }

    #[test]
    fn callback_cancels_when_stream_receiver_is_gone() {
        let (sender, receiver) = mpsc::unbounded_channel();
        drop(receiver);
        let mut context = CallbackContext {
            sender: Some(sender),
        };
        let event = event();
        let accepted =
            unsafe { stream_callback((&mut context as *mut CallbackContext).cast(), &event) };
        assert_eq!(accepted, 0);
    }

    #[test]
    fn callback_publishes_to_a_live_stream_receiver() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut context = CallbackContext {
            sender: Some(sender),
        };
        let event = event();
        let accepted =
            unsafe { stream_callback((&mut context as *mut CallbackContext).cast(), &event) };
        assert_eq!(accepted, 1);
        assert_eq!(receiver.try_recv().unwrap().token_id, 42);
    }
}
