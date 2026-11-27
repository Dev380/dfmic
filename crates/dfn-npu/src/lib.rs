//! The three DeepFilterNet3_ll graphs, driven one frame at a time on the NPU.
//!
//! Contract with `tools/dfnpy/convert.py` (which produces the IR):
//!   * every graph takes a fixed window of `frames` frames plus each GRU's
//!     output history, and returns outputs for the whole window plus the state
//!     after the newest frame;
//!   * only the **newest** frame of a windowed output is valid, because the
//!     causal convolutions need their predecessors as history;
//!   * each GRU advances exactly one frame per inference.
//!
//! So this type keeps two histories per graph — a ring of past input frames (for
//! the convolutions) and a ring of past GRU outputs (for the recurrence) — and
//! hands back one frame's worth of outputs per call.

pub mod ffi;

use std::collections::VecDeque;
use std::ffi::{c_void, CStr, CString};
use std::path::Path;

use dfn_dsp::{EncoderOutput, ModelRuntime, DF_ORDER, NB_DF, NB_ERB};
use num_complex::Complex32;

use crate::ffi::{
    cstring, ov_compiled_model_t, ov_core_t, ov_infer_request_t, ov_property, ov_tensor_t,
};

/// Per-graph shape of the conversion, mirroring `GRAPH_FRAMES` in convert.py.
struct GraphSpec {
    name: &'static str,
    frames: usize,
    /// Number of leading inputs that carry audio features (the rest are state).
    n_inputs: usize,
    /// Time axis of each of those inputs.
    input_time_axis: &'static [usize],
    /// Number of leading outputs that carry results (the rest are state).
    n_outputs: usize,
    /// Time axis of each of those outputs.
    output_time_axis: &'static [usize],
    /// Floats in one frame of each of those outputs.
    ///
    /// The DSP hard-codes all of these, so a model converted with a different
    /// ERB count, filter width or encoder width has to be rejected at load time:
    /// the alternative is a panic in `erb_mask` on the first hop, or a silently
    /// half-filled deep-filter tap set.
    output_widths: &'static [usize],
}

const ENC: GraphSpec = GraphSpec {
    name: "enc",
    frames: 6,
    n_inputs: 2,
    input_time_axis: &[2, 2],
    n_outputs: 7,
    output_time_axis: &[2, 2, 2, 2, 1, 2, 1],
    // e0, e1, e2, e3, emb, c0, lsnr. The skip connections keep their encoder
    // shapes ([1, 64, frames, 32], [1, 64, frames, 16], then 8 twice, giving
    // 2048, 1024, 512 and 512 floats per frame), `emb` is [1, frames, 512] and
    // `c0` is [1, 64, frames, 96]. The decoders are fed straight from these
    // buffers, so their widths are part of the contract too.
    output_widths: &[2048, 1024, 512, 512, 512, 64 * NB_DF, 1],
};

const ERB_DEC: GraphSpec = GraphSpec {
    name: "erb_dec",
    frames: 3,
    n_inputs: 5,
    input_time_axis: &[1, 2, 2, 2, 2],
    n_outputs: 1,
    output_time_axis: &[2],
    output_widths: &[NB_ERB],
};

const DF_DEC: GraphSpec = GraphSpec {
    name: "df_dec",
    frames: 3,
    n_inputs: 2,
    input_time_axis: &[1, 2],
    n_outputs: 2,
    output_time_axis: &[1, 1],
    // Complex taps interleaved as real/imaginary pairs, then the alpha value the
    // DSP ignores.
    output_widths: &[NB_DF * DF_ORDER * 2, 1],
};

fn product(dims: &[i64]) -> usize {
    dims.iter().product::<i64>() as usize
}

/// Borrow `len` elements at `ptr`, treating a null pointer as an empty array.
///
/// `slice::from_raw_parts` is undefined behaviour on a null pointer *even when
/// `len` is zero*, and OpenVINO leaves an out-parameter null when it has nothing
/// to hand back, so every array it returns goes through here. A null pointer
/// with a non-zero count is a broken library, not an empty array.
///
/// # Safety
/// A non-null `ptr` must point at `len` initialised `T`s that outlive the slice.
unsafe fn ffi_slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if ptr.is_null() {
        return if len == 0 { Some(&[]) } else { None };
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Owns an OpenVINO handle, freeing it however the scope is left.
///
/// Every port, tensor and compiled model in this file is handed to the caller, who
/// then makes several more calls that can fail. Without a guard each of those
/// `?` paths leaks the handle, and the trees behind ports and models are not
/// small.
struct Owned<T> {
    handle: *mut T,
    free: unsafe extern "C" fn(*mut T),
}

impl<T> Owned<T> {
    fn new(handle: *mut T, free: unsafe extern "C" fn(*mut T)) -> Self {
        Self { handle, free }
    }

    /// Give up ownership, for a caller that stores the handle itself.
    fn into_raw(self) -> *mut T {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.free)(self.handle) };
        }
    }
}

/// Write one frame's block into a windowed tensor at position `t` along `axis`.
///
/// Takes a raw pointer rather than a slice because the destination backs a
/// tensor created with `ov_tensor_create_from_host_ptr`: OpenVINO keeps that
/// pointer, and forming a `&mut [f32]` over the same memory afterwards would
/// invalidate it (the borrow would claim exclusive access that OpenVINO still
/// has). Every write to a tensor-backed buffer goes through a raw pointer for
/// that reason.
///
/// # Safety
/// `dest` must point at `product(dims)` initialised floats that no reference
/// aliases, and `src` must hold one frame of the same shape.
unsafe fn scatter_frame(dest: *mut f32, dims: &[i64], axis: usize, t: usize, src: &[f32]) {
    let block = product(&dims[axis + 1..]);
    let blocks = product(&dims[..axis]);
    let stride = block * dims[axis] as usize;
    for b in 0..blocks {
        let at = b * stride + t * block;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(b * block), dest.add(at), block);
        }
    }
}

/// Write `src` at `offset` in a tensor-backed buffer. See [`scatter_frame`].
///
/// # Safety
/// `dest` must point at an initialised buffer long enough for `offset + src.len()`
/// floats that no reference aliases.
unsafe fn write_block(dest: *mut f32, offset: usize, src: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest.add(offset), src.len());
    }
}

/// Read one frame's block out of a windowed tensor at position `t` along `axis`.
fn gather_frame(dest: &mut [f32], src: &[f32], dims: &[i64], axis: usize, t: usize) {
    let block = product(&dims[axis + 1..]);
    let blocks = product(&dims[..axis]);
    let stride = block * dims[axis] as usize;
    for b in 0..blocks {
        let at = b * stride + t * block;
        dest[b * block..(b + 1) * block].copy_from_slice(&src[at..at + block]);
    }
}

#[derive(Clone)]
struct Port {
    name: String,
    dims: Vec<i64>,
    elems: usize,
}

/// Wrap a host buffer in an input tensor.
///
/// The buffer must outlive the tensor and must never move or be resized, which
/// is why this is called once per input in `Graph::allocate` and why every later
/// write to those buffers goes through a raw pointer.
fn create_input_tensor(dims: &[i64], data: *mut f32) -> Result<*mut ov_tensor_t, String> {
    let mut shape = ffi::ov_shape {
        rank: dims.len() as i64,
        dims: std::ptr::null_mut(),
    };
    ffi::check("ov_shape_create", unsafe {
        ffi::ov_shape_create(
            dims.len() as i64,
            dims.as_ptr(),
            &mut shape as *mut _ as *mut ffi::ov_shape_t,
        )
    })?;
    let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
    let status = unsafe {
        ffi::ov_tensor_create_from_host_ptr(
            ffi::OV_ELEMENT_TYPE_F32,
            shape,
            data as *mut c_void,
            &mut tensor,
        )
    };
    // `ov_shape_free` reports a status, but the shape's dims are only ours to
    // release: there is nothing to retry and nothing better to do than carry on.
    let _ = unsafe { ffi::ov_shape_free(&mut shape as *mut _ as *mut ffi::ov_shape_t) };
    ffi::check("ov_tensor_create_from_host_ptr", status)?;
    Ok(tensor)
}

/// Attach an already-created input tensor to the request.
///
/// The name is a cached `CString` owned by the graph: building one per input of
/// every graph on every hop was a hidden allocation 100 times a second, which is
/// the exact cost the cached tensors exist to avoid.
fn attach_input(
    req: *mut ov_infer_request_t,
    name: &CStr,
    tensor: *mut ov_tensor_t,
) -> Result<(), String> {
    ffi::check("ov_infer_request_set_tensor", unsafe {
        ffi::ov_infer_request_set_tensor(req, name.as_ptr(), tensor)
    })
    .map_err(String::from)
}

/// Read output tensor `index` into `dest`, keeping only the newest frame of a
/// windowed output (or the whole tensor when `time_axis` is `None`).
fn read_output(
    req: *mut ov_infer_request_t,
    index: usize,
    dims: &[i64],
    time_axis: Option<usize>,
    frames: usize,
    dest: &mut [f32],
    debug_raw: bool,
) -> Result<(), String> {
    let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
    ffi::check("ov_infer_request_get_output_tensor_by_index", unsafe {
        ffi::ov_infer_request_get_output_tensor_by_index(req, index, &mut tensor)
    })?;
    // The request hands out a fresh handle per call and keeps none of them, so
    // every `?` below would otherwise leak it.
    let tensor = Owned::new(tensor, ffi::ov_tensor_free);

    // Reading the buffer as `f32` is only sound if it *is* f32 and long enough
    // for the element count: an f16 or bf16 output reinterpreted as f32 walks
    // twice its length past the end of the allocation.
    let mut element_type = -1i32;
    ffi::check("ov_tensor_get_element_type", unsafe {
        ffi::ov_tensor_get_element_type(tensor.handle, &mut element_type)
    })?;
    if element_type != ffi::OV_ELEMENT_TYPE_F32 {
        return Err(format!(
            "output {index}: element type {element_type}, expected f32 ({})",
            ffi::OV_ELEMENT_TYPE_F32
        ));
    }
    let mut elems = 0usize;
    ffi::check("ov_tensor_get_size", unsafe {
        ffi::ov_tensor_get_size(tensor.handle, &mut elems)
    })?;
    let mut bytes = 0usize;
    ffi::check("ov_tensor_get_byte_size", unsafe {
        ffi::ov_tensor_get_byte_size(tensor.handle, &mut bytes)
    })?;
    let needed = elems * std::mem::size_of::<f32>();
    if bytes < needed {
        return Err(format!(
            "output {index}: {elems} f32 elements need {needed} bytes, tensor has {bytes}"
        ));
    }

    let mut ptr: *mut c_void = std::ptr::null_mut();
    ffi::check("ov_tensor_data", unsafe {
        ffi::ov_tensor_data(tensor.handle, &mut ptr)
    })?;
    let src = unsafe { ffi_slice(ptr as *const f32, elems) }.ok_or_else(|| {
        format!("output {index}: OpenVINO returned a null pointer for {elems} elements")
    })?;
    if debug_raw {
        // No dims here: for a state output the caller passes the *input* port's
        // shape, which is a different rank, and printing it would be a lie.
        eprintln!(
            "RAW out[{index}] elems={elems} first={:?}",
            &src[..elems.min(8)]
        );
    }

    match time_axis {
        Some(axis) => gather_frame(dest, src, dims, axis, frames - 1),
        None => {
            let n = dest.len().min(elems);
            dest[..n].copy_from_slice(&src[..n]);
        }
    }
    Ok(())
}

/// One converted graph with its state and history.
pub struct Graph {
    cm: *mut ov_compiled_model_t,
    req: *mut ov_infer_request_t,
    spec: &'static GraphSpec,
    inputs: Vec<Port>,
    /// State inputs, sized per frame (not per window).
    states: Vec<Port>,
    outputs: Vec<Port>,
    /// (output index, state index) for each state output.
    state_outs: Vec<(usize, usize)>,
    /// Reusable host buffers for building inputs.
    ///
    /// These own the memory; `in_ptr` is how it is written. Every write to a
    /// tensor-backed buffer goes through the raw pointer, and the `Vec` is only
    /// used to allocate and to free, because forming a `&mut [f32]` over the same
    /// memory after `ov_tensor_create_from_host_ptr` would invalidate the pointer
    /// OpenVINO is holding.
    in_buf: Vec<Vec<f32>>,
    in_ptr: Vec<*mut f32>,
    /// Past frames per input, oldest first, always exactly `frames` long.
    in_hist: Vec<VecDeque<Vec<f32>>>,
    /// Past GRU outputs per state, oldest first, always `frames - 1` long.
    state_hist: Vec<VecDeque<Vec<f32>>>,
    out_buf: Vec<Vec<f32>>,
    /// Flattened GRU state history, rebuilt in place each hop. Owned here rather
    /// than allocated per hop because its address must stay put: see
    /// `in_tensors`.
    state_flat: Vec<Vec<f32>>,
    /// See `in_ptr`.
    state_ptr: Vec<*mut f32>,
    /// Input tensors, created once and reused.
    ///
    /// `ov_tensor_create_from_host_ptr` wraps *our* buffer, so a tensor stays
    /// valid as long as that buffer does not move. Every buffer it points at is
    /// sized in `allocate` and only ever written in place, so the tensors are
    /// created there and only re-attached to the request per hop. This removes
    /// a shape allocation, a tensor allocation and a tensor free for every input
    /// of every graph, 100 times a second.
    in_tensors: Vec<*mut ov_tensor_t>,
    /// Same, for the stacked state-history inputs.
    state_tensors: Vec<*mut ov_tensor_t>,
    /// Input names as C strings, converted once for the same reason as the
    /// tensors: `attach_input` needs a `*const c_char` on every hop, and
    /// `CString::new` would allocate a fresh copy of every name every time.
    in_names: Vec<CString>,
    state_names: Vec<CString>,
    /// `DFMIC_DEBUG_RAW` was set: print every output tensor.
    debug_raw: bool,
    pub infer_count: u64,
}

impl Graph {
    fn compile(
        core: *mut ov_core_t,
        xml: &Path,
        spec: &'static GraphSpec,
        device: &str,
        precision: &str,
    ) -> Result<Self, String> {
        let path = cstring(&xml.to_string_lossy());
        let dev = cstring(device);
        let key = cstring("INFERENCE_PRECISION_HINT");
        let val = cstring(precision);
        let props = [ov_property {
            key: key.as_ptr(),
            value: val.as_ptr() as *const c_void,
        }];
        let mut cm: *mut ov_compiled_model_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::ov_core_compile_model_from_file_props(
                core,
                path.as_ptr(),
                dev.as_ptr(),
                1,
                props.as_ptr(),
                &mut cm,
            )
        };
        ffi::check("ov_core_compile_model_from_file_props", status)
            .map_err(|e| format!("compiling {} on {}: {e}", spec.name, device))?;

        let mut req: *mut ov_infer_request_t = std::ptr::null_mut();
        // The compiled model is owned here rather than straight into a `Graph`
        // because `Graph` does not exist yet: if creating the request fails, this
        // guard is the only thing that will free a fully compiled NPU model.
        let cm = Owned::new(cm, ffi::ov_compiled_model_free);
        ffi::check("ov_compiled_model_create_infer_request", unsafe {
            ffi::ov_compiled_model_create_infer_request(cm.handle, &mut req)
        })
        .map_err(|e| format!("creating an infer request for {}: {e}", spec.name))?;

        let mut graph = Self {
            cm: cm.into_raw(),
            req,
            spec,
            inputs: Vec::new(),
            states: Vec::new(),
            outputs: Vec::new(),
            state_outs: Vec::new(),
            in_buf: Vec::new(),
            in_ptr: Vec::new(),
            in_hist: Vec::new(),
            state_hist: Vec::new(),
            out_buf: Vec::new(),
            state_flat: Vec::new(),
            state_ptr: Vec::new(),
            in_tensors: Vec::new(),
            state_tensors: Vec::new(),
            in_names: Vec::new(),
            state_names: Vec::new(),
            // Read once: `std::env::var` locks the environment and allocates,
            // which is far too expensive for something read on every output of
            // every hop.
            debug_raw: std::env::var_os("DFMIC_DEBUG_RAW").is_some(),
            infer_count: 0,
        };
        // From here on `graph` owns the request and the model, so a failure drops
        // it and `Drop` releases both along with any tensors already created.
        // Every handle starts null and both tensor vectors are drained, so a
        // half-built graph is safe to drop.
        graph.discover_ports()?;
        graph.allocate()?;
        Ok(graph)
    }

    fn discover_ports(&mut self) -> Result<(), String> {
        let mut n_in = 0usize;
        let mut n_out = 0usize;
        ffi::check("ov_compiled_model_inputs_size", unsafe {
            ffi::ov_compiled_model_inputs_size(self.cm, &mut n_in)
        })?;
        ffi::check("ov_compiled_model_outputs_size", unsafe {
            ffi::ov_compiled_model_outputs_size(self.cm, &mut n_out)
        })?;

        for i in 0..n_in {
            self.inputs.push(self.input_port(i)?);
        }
        for i in 0..n_out {
            self.outputs.push(self.output_port(i)?);
        }

        // The converter appends state tensors after the feature tensors, so the
        // first `n_inputs` inputs are features and the rest are state.
        if self.inputs.len() < self.spec.n_inputs || self.outputs.len() < self.spec.n_outputs {
            return Err(format!(
                "{}: expected at least {} inputs and {} outputs, model has {} and {}",
                self.spec.name,
                self.spec.n_inputs,
                self.spec.n_outputs,
                self.inputs.len(),
                self.outputs.len()
            ));
        }

        let state_inputs: Vec<Port> = self.inputs.drain(self.spec.n_inputs..).collect();
        let state_outputs: Vec<Port> = self.outputs.drain(self.spec.n_outputs..).collect();
        if state_inputs.len() != state_outputs.len() {
            return Err(format!(
                "{}: {} state inputs but {} state outputs",
                self.spec.name,
                state_inputs.len(),
                state_outputs.len()
            ));
        }

        // State inputs are [frames - 1, 1, H]; the matching outputs are [1, H].
        let mut states = Vec::new();
        for (i, port) in state_inputs.iter().enumerate() {
            if port.elems % (self.spec.frames - 1) != 0 {
                return Err(format!(
                    "{}: state input {} has {} elements, not divisible by {}",
                    self.spec.name,
                    port.name,
                    port.elems,
                    self.spec.frames - 1
                ));
            }
            let per_frame = port.elems / (self.spec.frames - 1);
            if per_frame != state_outputs[i].elems {
                return Err(format!(
                    "{}: state {} has {per_frame} elements per frame but its output has {}",
                    self.spec.name, port.name, state_outputs[i].elems
                ));
            }
            states.push(Port {
                name: port.name.clone(),
                dims: port.dims.clone(),
                elems: per_frame,
            });
            self.state_outs.push((self.spec.n_outputs + i, i));
        }
        self.states = states;

        // Validate the window shape, so a converter change cannot silently corrupt
        // audio by feeding misaligned frames.
        for (i, port) in self.inputs.iter().enumerate() {
            let axis = self.spec.input_time_axis[i];
            if port.dims.get(axis).copied() != Some(self.spec.frames as i64) {
                return Err(format!(
                    "{}: input {} has shape {:?}, expected axis {axis} to be {}",
                    self.spec.name, port.name, port.dims, self.spec.frames
                ));
            }
        }
        for (i, port) in self.outputs.iter().enumerate() {
            let axis = self.spec.output_time_axis[i];
            if port.dims.get(axis).copied() != Some(self.spec.frames as i64) {
                return Err(format!(
                    "{}: output {} has shape {:?}, expected axis {axis} to be {}",
                    self.spec.name, port.name, port.dims, self.spec.frames
                ));
            }
        }
        // Validate the per-frame widths too. The DSP reads these outputs by
        // width — `erb_mask` copies NB_ERB floats, `df_coefs` reads one complex
        // pair per tap — so a model converted for a different filterbank would
        // panic at best and hand back stale coefficients at worst. Both are
        // unrecoverable at run time, so reject the model at load.
        for (i, port) in self.outputs.iter().enumerate() {
            let want = self.spec.output_widths[i];
            let got = self.per_frame(i, port);
            if got != want {
                return Err(format!(
                    "{}: output {} has {got} elements per frame (shape {:?}), the DSP needs {want}",
                    self.spec.name, port.name, port.dims
                ));
            }
        }
        Ok(())
    }

    fn port_at(&self, index: usize, output: bool) -> Result<Port, String> {
        let mut port_ptr = std::ptr::null_mut();
        if output {
            ffi::check("ov_compiled_model_output_by_index", unsafe {
                ffi::ov_compiled_model_output_by_index(self.cm, index, &mut port_ptr)
            })?;
        } else {
            ffi::check("ov_compiled_model_input_by_index", unsafe {
                ffi::ov_compiled_model_input_by_index(self.cm, index, &mut port_ptr)
            })?;
        }
        // A fresh port per call, owned by us: leaking it on a failed shape or
        // name lookup would leak a whole output node per attempt.
        let port = Owned::new(port_ptr, ffi::ov_output_const_port_free);
        unsafe { read_port(port.handle) }
    }

    fn input_port(&self, index: usize) -> Result<Port, String> {
        self.port_at(index, false)
    }

    fn output_port(&self, index: usize) -> Result<Port, String> {
        self.port_at(index, true)
    }

    /// Elements in one frame of output `i`.
    fn per_frame(&self, i: usize, port: &Port) -> usize {
        let axis = self.spec.output_time_axis[i];
        let mut n = 1usize;
        for (d, size) in port.dims.iter().enumerate() {
            if d != axis {
                n *= *size as usize;
            }
        }
        n
    }

    fn allocate(&mut self) -> Result<(), String> {
        self.in_buf = self.inputs.iter().map(|p| vec![0.0f32; p.elems]).collect();
        self.in_hist = self
            .inputs
            .iter()
            .map(|p| {
                VecDeque::from(vec![
                    vec![0.0f32; p.elems / self.spec.frames];
                    self.spec.frames
                ])
            })
            .collect();
        self.state_hist = self
            .states
            .iter()
            .map(|p| VecDeque::from(vec![vec![0.0f32; p.elems]; self.spec.frames - 1]))
            .collect();
        // Only the newest frame of a windowed output is valid, so the output
        // buffers hold exactly one frame: sizing them for the whole window would
        // hand the next graph a buffer with stale data past the valid block.
        // Sized once, never resized: these back the cached input tensors.
        self.state_flat = self
            .states
            .iter()
            .map(|p| vec![0.0f32; p.elems * (self.spec.frames - 1)])
            .collect();
        // Take the pointers once, and write through them from here on: see the
        // note on `in_buf`. A `Vec` never moves its buffer, so these stay valid
        // for the graph's life.
        self.in_ptr = self.in_buf.iter_mut().map(Vec::as_mut_ptr).collect();
        self.state_ptr = self.state_flat.iter_mut().map(Vec::as_mut_ptr).collect();
        self.out_buf = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, p)| vec![0.0f32; self.per_frame(i, p)])
            .collect();

        // Tensors and names go into the graph's own vectors as they are created,
        // so a failure part way through leaves the earlier ones to `Drop`: the
        // caller propagates the error, drops the half-built graph and every
        // handle it already holds goes with it.
        self.in_names = self
            .inputs
            .iter()
            .map(|p| CString::new(p.name.as_str()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: input name is not a C string: {e}", self.spec.name))?;
        self.state_names = self
            .states
            .iter()
            .map(|p| CString::new(p.name.as_str()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: state name is not a C string: {e}", self.spec.name))?;
        for i in 0..self.inputs.len() {
            let tensor = create_input_tensor(&self.inputs[i].dims, self.in_ptr[i])?;
            self.in_tensors.push(tensor);
        }
        for i in 0..self.states.len() {
            let tensor = create_input_tensor(&self.states[i].dims, self.state_ptr[i])?;
            self.state_tensors.push(tensor);
        }
        Ok(())
    }

    /// Forget the history without recompiling or reallocating.
    ///
    /// Zeroing in place is not an optimisation, it is a requirement: the buffers
    /// below back tensors created with `ov_tensor_create_from_host_ptr`, so
    /// replacing them would leave OpenVINO pointing at freed memory. Only the
    /// values may change.
    fn reset(&mut self) {
        // Through the raw pointers, for the same reason `push` does: these
        // buffers back cached tensors.
        for (buffer, pointer) in self.in_buf.iter().zip(&self.in_ptr) {
            unsafe { std::ptr::write_bytes(*pointer, 0, buffer.len()) };
        }
        for (buffer, pointer) in self.state_flat.iter().zip(&self.state_ptr) {
            unsafe { std::ptr::write_bytes(*pointer, 0, buffer.len()) };
        }
        // These two do not back tensors, so a slice is fine.
        for buffer in &mut self.out_buf {
            buffer.fill(0.0);
        }
        for history in &mut self.in_hist {
            for frame in history.iter_mut() {
                frame.fill(0.0);
            }
        }
        for history in &mut self.state_hist {
            for state in history.iter_mut() {
                state.fill(0.0);
            }
        }
        self.infer_count = 0;
    }

    /// Push one frame through the graph. `frame` holds one block per feature input.
    fn push(&mut self, frame: &[&[f32]]) -> Result<(), String> {
        if frame.len() != self.inputs.len() {
            return Err(format!(
                "{}: {} inputs, got {}",
                self.spec.name,
                self.inputs.len(),
                frame.len()
            ));
        }
        for (i, value) in frame.iter().enumerate() {
            let expected = self.in_hist[i].front().map_or(0, Vec::len);
            if value.len() != expected {
                return Err(format!(
                    "{}: input {} is {} long, the graph wants {expected}",
                    self.spec.name,
                    self.inputs[i].name,
                    value.len()
                ));
            }
        }

        for (i, value) in frame.iter().enumerate() {
            // Recycle the frame leaving the window instead of dropping it: this
            // runs 100 times a second for every input of every graph.
            let mut recycled = self.in_hist[i].pop_front().expect("history is primed");
            recycled.copy_from_slice(value);
            self.in_hist[i].push_back(recycled);
        }

        // Build the windowed feature tensors from the frame histories. Read the
        // history in place: cloning it first copied every frame of every input
        // on every hop, for no reason. The buffers are written through raw
        // pointers because they back cached tensors: see `scatter_frame`.
        for i in 0..self.inputs.len() {
            let axis = self.spec.input_time_axis[i];
            let dims = &self.inputs[i].dims;
            let dest = self.in_ptr[i];
            for (t, frame) in self.in_hist[i].iter().enumerate() {
                unsafe { scatter_frame(dest, dims, axis, t, frame) };
            }
        }

        // Stack the GRU state history into its [frames - 1, 1, H] tensor, in
        // place: the buffer backs a cached tensor, so it must not move.
        for (i, history) in self.state_hist.iter().enumerate() {
            let dest = self.state_ptr[i];
            let mut at = 0;
            for state in history.iter() {
                unsafe { write_block(dest, at, state) };
                at += state.len();
            }
        }

        let req = self.req;
        // Tensors and names were built once in `allocate`; only re-attach them.
        // Nothing here allocates.
        for i in 0..self.inputs.len() {
            attach_input(req, &self.in_names[i], self.in_tensors[i])?;
        }
        for i in 0..self.states.len() {
            attach_input(req, &self.state_names[i], self.state_tensors[i])?;
        }

        let status = unsafe { ffi::ov_infer_request_infer(req) };
        ffi::check("ov_infer_request_infer", status)?;
        self.infer_count += 1;

        // Read results: newest frame of windowed outputs, whole state tensors.
        for i in 0..self.outputs.len() {
            let dims = &self.outputs[i].dims;
            let axis = self.spec.output_time_axis[i];
            let mut buf = std::mem::take(&mut self.out_buf[i]);
            let res = read_output(
                req,
                i,
                dims,
                Some(axis),
                self.spec.frames,
                &mut buf,
                self.debug_raw,
            );
            self.out_buf[i] = buf;
            res?;
        }
        for (out_index, state_index) in self.state_outs.iter().copied() {
            let dims = &self.states[state_index].dims;
            // Reuse the allocation of the state leaving the window.
            let mut next = self.state_hist[state_index]
                .pop_front()
                .expect("history is primed");
            let res = read_output(req, out_index, dims, None, 1, &mut next, self.debug_raw);
            self.state_hist[state_index].push_back(next);
            res?;
        }
        Ok(())
    }

    /// Newest frame of output `i`.
    pub fn output(&self, i: usize) -> &[f32] {
        &self.out_buf[i]
    }
}

// OpenVINO's C API has no thread affinity: these objects must not be *used*
// concurrently, but moving ownership to another thread is fine, which is exactly
// what the daemon does when it hands the pipeline to its audio worker.
unsafe impl Send for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            // Tensors before the request: they are still attached to it.
            for tensor in self
                .in_tensors
                .drain(..)
                .chain(self.state_tensors.drain(..))
            {
                if !tensor.is_null() {
                    ffi::ov_tensor_free(tensor);
                }
            }
            if !self.req.is_null() {
                ffi::ov_infer_request_free(self.req);
            }
            if !self.cm.is_null() {
                ffi::ov_compiled_model_free(self.cm);
            }
        }
    }
}

unsafe fn read_port(port: *const ffi::ov_output_const_port_t) -> Result<Port, String> {
    let mut shape = ffi::ov_shape {
        rank: 0,
        dims: std::ptr::null_mut(),
    };
    ffi::check("ov_const_port_get_shape", unsafe {
        ffi::ov_const_port_get_shape(port, &mut shape as *mut _ as *mut ffi::ov_shape_t)
    })?;
    let rank = shape.rank as usize;
    // Freed on both paths: a successful call may still return dims the caller has
    // to free, so the early return below must not skip this.
    let dims = unsafe { ffi_slice(shape.dims, rank) }.map(<[i64]>::to_vec);
    // See `create_input_tensor`: the status is not actionable.
    let _ = unsafe { ffi::ov_shape_free(&mut shape as *mut _ as *mut ffi::ov_shape_t) };
    let dims =
        dims.ok_or_else(|| format!("port shape has rank {rank} but a null dimensions array"))?;

    let mut name_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
    let name = if unsafe { ffi::ov_port_get_any_name(port, &mut name_ptr) } == ffi::OK
        && !name_ptr.is_null()
    {
        let s = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::ov_free(name_ptr) };
        s
    } else {
        String::from("<unnamed>")
    };

    Ok(Port {
        name,
        elems: product(&dims),
        dims,
    })
}

/// All three graphs plus the owning OpenVINO core.
pub struct NpuRuntime {
    core: *mut ov_core_t,
    enc: Graph,
    erb_dec: Graph,
    df_dec: Graph,
    pub device: String,
}

impl NpuRuntime {
    /// Load the converted IR from `ir_dir` and compile it for `device`.
    ///
    /// `cache_dir`, when given, lets OpenVINO cache compiled blobs so a service
    /// restart skips the multi-second NPU compile.
    pub fn load(ir_dir: &Path, device: &str, cache_dir: Option<&Path>) -> Result<Self, String> {
        let mut core: *mut ov_core_t = std::ptr::null_mut();
        ffi::check("ov_core_create", unsafe { ffi::ov_core_create(&mut core) })?;

        if let Some(dir) = cache_dir {
            let dir_s = cstring(&dir.to_string_lossy());
            let dev = cstring(device);
            let k_cache = cstring("CACHE_DIR");
            let v_mode = cstring("OPTIMIZE_SPEED");
            let k_mode = cstring("CACHE_MODE");
            let props = [
                ov_property {
                    key: k_cache.as_ptr(),
                    value: dir_s.as_ptr() as *const c_void,
                },
                ov_property {
                    key: k_mode.as_ptr(),
                    value: v_mode.as_ptr() as *const c_void,
                },
            ];
            let status =
                unsafe { ffi::ov_core_set_properties(core, dev.as_ptr(), 2, props.as_ptr()) };
            if let Err(e) = ffi::check("ov_core_set_properties", status) {
                // Caching is an optimisation; never fail the daemon over it.
                eprintln!("dfmic: warning: could not enable model caching: {e}");
            }
        }

        let loaded = (|| {
            let enc = Graph::compile(core, &ir_dir.join("enc.xml"), &ENC, device, "f16")?;
            let erb_dec =
                Graph::compile(core, &ir_dir.join("erb_dec.xml"), &ERB_DEC, device, "f16")?;
            let df_dec = Graph::compile(core, &ir_dir.join("df_dec.xml"), &DF_DEC, device, "f16")?;
            Ok((enc, erb_dec, df_dec))
        })();
        match loaded {
            Ok((enc, erb_dec, df_dec)) => Ok(Self {
                core,
                enc,
                erb_dec,
                df_dec,
                device: device.to_string(),
            }),
            // A failed load has to release the core itself: nothing else owns
            // it yet, and a leaked core keeps its NPU context alive, so the next
            // attempt (or another program) finds the device busy.
            Err(error) => {
                unsafe { ffi::ov_core_free(core) };
                Err(error)
            }
        }
    }

    /// Forget every graph's history, for a stream that restarted.
    pub fn reset(&mut self) {
        self.enc.reset();
        self.erb_dec.reset();
        self.df_dec.reset();
    }

    /// Total inferences performed, for status reporting.
    pub fn infer_count(&self) -> u64 {
        self.enc.infer_count
    }
}

unsafe impl Send for NpuRuntime {}

impl Drop for NpuRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.core.is_null() {
                ffi::ov_core_free(self.core);
            }
        }
    }
}

/// Lists devices OpenVINO can see, for `dfmic doctor`.
pub fn available_devices() -> Result<Vec<String>, String> {
    let mut core: *mut ov_core_t = std::ptr::null_mut();
    ffi::check("ov_core_create", unsafe { ffi::ov_core_create(&mut core) })?;
    let mut devices = ffi::ov_available_devices {
        devices: std::ptr::null_mut(),
        size: 0,
    };
    let status = unsafe { ffi::ov_core_get_available_devices(core, &mut devices) };
    // Read the names out before freeing them, and free the list and the core on
    // every path: `?` here would leak both.
    let out = ffi::check("ov_core_get_available_devices", status)
        .map_err(String::from)
        .and_then(|()| {
            let names = unsafe { ffi_slice(devices.devices, devices.size) }.ok_or_else(|| {
                format!(
                    "OpenVINO reported {} devices but returned a null array",
                    devices.size
                )
            })?;
            names
                .iter()
                .map(|p| {
                    if p.is_null() {
                        return Err("OpenVINO returned a null device name".to_string());
                    }
                    Ok(unsafe { CStr::from_ptr(*p) }.to_string_lossy().into_owned())
                })
                .collect::<Result<Vec<String>, String>>()
        });
    unsafe {
        ffi::ov_available_devices_free(&mut devices);
        ffi::ov_core_free(core);
    }
    out
}

impl ModelRuntime for NpuRuntime {
    fn encode_into(&mut self, feat_erb: &[f32], feat_spec: &[f32], out: &mut EncoderOutput) {
        self.enc
            .push(&[feat_erb, feat_spec])
            .expect("encoder inference failed");
        // Copy into the caller's buffers rather than allocating six new ones,
        // sizing each from its own output: the skip connections are not the same
        // length as `e0` (2048, 1024, 512 and 512 elements per frame), so one
        // length for all of them would silently mis-shape four of the six.
        // The widths were pinned in `discover_ports`, so `copy_from_slice`
        // cannot panic here.
        for (slot, index) in [
            (&mut out.e0, 0),
            (&mut out.e1, 1),
            (&mut out.e2, 2),
            (&mut out.e3, 3),
            (&mut out.emb, 4),
            (&mut out.c0, 5),
        ] {
            let src = self.enc.output(index);
            if slot.len() != src.len() {
                slot.resize(src.len(), 0.0);
            }
            slot.copy_from_slice(src);
        }
        out.lsnr = self.enc.output(6)[0];
    }

    fn erb_mask(&mut self, enc: &EncoderOutput) -> [f32; NB_ERB] {
        self.erb_dec
            .push(&[&enc.emb, &enc.e3, &enc.e2, &enc.e1, &enc.e0])
            .expect("erb decoder inference failed");
        let src = self.erb_dec.output(0);
        assert_eq!(
            src.len(),
            NB_ERB,
            "erb decoder produced {} gains, not {NB_ERB}",
            src.len()
        );
        let mut gains = [0.0f32; NB_ERB];
        gains.copy_from_slice(src);
        gains
    }

    fn df_coefs(&mut self, enc: &EncoderOutput, out: &mut [Complex32; NB_DF * DF_ORDER]) {
        self.df_dec
            .push(&[&enc.emb, &enc.c0])
            .expect("df decoder inference failed");
        // Pairs of real/imaginary values, one per tap, laid out band-major. A
        // short output would leave the tail of `out` holding the previous hop's
        // coefficients and quietly filter with them, so the width is checked
        // rather than zipped short.
        let src = self.df_dec.output(0);
        assert_eq!(
            src.len(),
            NB_DF * DF_ORDER * 2,
            "df decoder produced {} coefficients, not {}",
            src.len(),
            NB_DF * DF_ORDER * 2
        );
        for (coefficient, pair) in out.iter_mut().zip(src.as_chunks::<2>().0) {
            *coefficient = Complex32::new(pair[0], pair[1]);
        }
    }

    fn reset(&mut self) {
        NpuRuntime::reset(self);
    }
}
