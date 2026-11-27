//! Minimal, hand-written bindings to OpenVINO's C API.
//!
//! `openvino-rs` tracks specific OpenVINO releases and breaks whenever Intel
//! changes the C ABI, while nixpkgs ships whatever it ships. The surface we need
//! is small — create a core, compile a model from an IR file with properties,
//! create an infer request, set tensors, infer, read tensors — so we bind exactly
//! that and nothing else.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_void, CString};

pub enum ov_core_t {}
pub enum ov_compiled_model_t {}
pub enum ov_infer_request_t {}
pub enum ov_tensor_t {}
pub enum ov_output_const_port_t {}

/// `ov_shape_t`: rank plus a pointer to the dimensions.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ov_shape {
    pub rank: i64,
    pub dims: *mut i64,
}

/// The C typedef spells the same struct `ov_shape_t` in function signatures.
pub type ov_shape_t = ov_shape;

#[repr(C)]
pub struct ov_property {
    pub key: *const c_char,
    pub value: *const c_void,
}

#[repr(C)]
pub struct ov_available_devices {
    pub devices: *mut *mut c_char,
    pub size: usize,
}

/// `ov_element_type_e`, only the variants we use. C enums are `int`-wide, which
/// is why the element type travels as an `i32` in the signatures below.
pub const OV_ELEMENT_TYPE_F32: i32 = 4;

pub type ov_status = i32;
pub const OK: ov_status = 0;

#[link(name = "openvino_c")]
extern "C" {
    pub fn ov_core_create(core: *mut *mut ov_core_t) -> ov_status;
    pub fn ov_core_free(core: *mut ov_core_t);
    pub fn ov_core_set_properties(
        core: *const ov_core_t,
        device_name: *const c_char,
        num_properties: usize,
        properties: *const ov_property,
    ) -> ov_status;
    pub fn ov_core_compile_model_from_file_props(
        core: *const ov_core_t,
        model_path: *const c_char,
        device_name: *const c_char,
        num_properties: usize,
        properties: *const ov_property,
        compiled_model: *mut *mut ov_compiled_model_t,
    ) -> ov_status;
    pub fn ov_core_get_available_devices(
        core: *const ov_core_t,
        devices: *mut ov_available_devices,
    ) -> ov_status;
    pub fn ov_available_devices_free(devices: *mut ov_available_devices);

    pub fn ov_compiled_model_free(compiled_model: *mut ov_compiled_model_t);
    pub fn ov_compiled_model_create_infer_request(
        compiled_model: *const ov_compiled_model_t,
        infer_request: *mut *mut ov_infer_request_t,
    ) -> ov_status;
    pub fn ov_compiled_model_inputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> ov_status;
    pub fn ov_compiled_model_outputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> ov_status;
    pub fn ov_compiled_model_input_by_index(
        compiled_model: *const ov_compiled_model_t,
        index: usize,
        input_port: *mut *mut ov_output_const_port_t,
    ) -> ov_status;
    pub fn ov_compiled_model_output_by_index(
        compiled_model: *const ov_compiled_model_t,
        index: usize,
        output_port: *mut *mut ov_output_const_port_t,
    ) -> ov_status;

    pub fn ov_const_port_get_shape(
        port: *const ov_output_const_port_t,
        shape: *mut ov_shape,
    ) -> ov_status;
    pub fn ov_port_get_any_name(
        port: *const ov_output_const_port_t,
        name: *mut *mut c_char,
    ) -> ov_status;
    pub fn ov_output_const_port_free(port: *mut ov_output_const_port_t);
    pub fn ov_free(content: *const c_char);

    pub fn ov_infer_request_infer(infer_request: *mut ov_infer_request_t) -> ov_status;
    pub fn ov_infer_request_set_tensor(
        infer_request: *mut ov_infer_request_t,
        tensor_name: *const c_char,
        tensor: *const ov_tensor_t,
    ) -> ov_status;
    pub fn ov_infer_request_get_output_tensor_by_index(
        infer_request: *const ov_infer_request_t,
        index: usize,
        tensor: *mut *mut ov_tensor_t,
    ) -> ov_status;
    pub fn ov_infer_request_free(infer_request: *mut ov_infer_request_t);

    pub fn ov_tensor_create_from_host_ptr(
        element_type: i32,
        shape: ov_shape,
        host_ptr: *mut c_void,
        tensor: *mut *mut ov_tensor_t,
    ) -> ov_status;
    pub fn ov_tensor_data(tensor: *const ov_tensor_t, data: *mut *mut c_void) -> ov_status;
    pub fn ov_tensor_get_size(tensor: *const ov_tensor_t, size: *mut usize) -> ov_status;
    /// `size` counts elements; `ov_tensor_get_byte_size` is the one that bounds
    /// a reinterpretation of the buffer.
    pub fn ov_tensor_get_byte_size(tensor: *const ov_tensor_t, size: *mut usize) -> ov_status;
    pub fn ov_tensor_get_element_type(
        tensor: *const ov_tensor_t,
        element_type: *mut i32,
    ) -> ov_status;
    pub fn ov_tensor_free(tensor: *mut ov_tensor_t);

    pub fn ov_shape_create(rank: i64, dims: *const i64, shape: *mut ov_shape) -> ov_status;
    pub fn ov_shape_free(shape: *mut ov_shape) -> ov_status;
}

/// An OpenVINO error, including the status code so callers can report it.
#[derive(Debug)]
pub struct OvError {
    pub what: &'static str,
    pub status: ov_status,
}

impl std::fmt::Display for OvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OpenVINO call {} failed with status {}",
            self.what, self.status
        )
    }
}

impl std::error::Error for OvError {}

impl From<OvError> for String {
    fn from(e: OvError) -> String {
        e.to_string()
    }
}

pub type Result<T> = std::result::Result<T, OvError>;

pub fn check(what: &'static str, status: ov_status) -> Result<()> {
    if status == OK {
        Ok(())
    } else {
        Err(OvError { what, status })
    }
}

/// A property value that lives long enough for the compile call.
pub fn cstring(s: &str) -> CString {
    CString::new(s).expect("no interior NUL in OpenVINO strings")
}
